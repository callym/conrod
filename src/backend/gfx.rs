//! A gfx-rs backend for rendering conrod primitives.

pub use gfx;
pub use glutin;

use gfx::{ texture };
use gfx::traits::FactoryExt;
use gfx_core;
use gfx_window_glutin;
use gfx_device_gl::{ CommandBuffer, Device, Factory, Resources };

use {Rect, Scalar};
use color;
use image;
use render;
use std;
use text;
use text::rt;

/// Draw text from the text cache texture `tex` in the fragment shader.
pub const MODE_TEXT: u32 = 0;
/// Draw an image from the texture at `tex` in the fragment shader.
pub const MODE_IMAGE: u32 = 1;
/// Ignore `tex` and draw simple, colored 2D geometry.
pub const MODE_GEOMETRY: u32 = 2;

const FRAGMENT_SHADER: &'static [u8] = b"
	#version 140

	uniform sampler2D t_Color;

	in vec2 v_Uv;
	in vec4 v_Color;
	flat in uint v_Mode;

	out vec4 f_Color;

	void main() {
        if (v_Mode == uint(0)) {
			// Text
    		f_Color = v_Color * vec4(1.0, 1.0, 1.0, texture(t_Color, v_Uv).a);
        } else if (v_Mode == uint(1)) {
			// Image
            f_Color = texture(t_Color, v_Uv);
        } else if (v_Mode == uint(2)) {
			// 2D Geometry
            f_Color = v_Color;
        }
	}
";

const VERTEX_SHADER: &'static [u8] = b"
	#version 140

	in vec2 a_Pos;
	in vec2 a_Uv;
	in vec4 a_Color;
	in uint a_Mode;

	out vec2 v_Uv;
	out vec4 v_Color;
	flat out uint v_Mode;

	void main() {
		v_Uv = a_Uv;
		v_Color = a_Color;
		v_Mode = a_Mode;
		gl_Position = vec4(a_Pos, 0.0, 1.0);
	}
";

// Format definitions (must be pub for  gfx_defines to use them)
pub type ColorFormat = gfx::format::Srgba8;
type DepthFormat = gfx::format::DepthStencil;
type SurfaceFormat = gfx::format::R8_G8_B8_A8;
type FullFormat = (SurfaceFormat, gfx::format::Unorm);

#[allow(unsafe_code)]
mod gfx_impl {
	use gfx;
	use super::ColorFormat;
	// Vertex and pipeline declarations
	gfx_defines! {
		vertex Vertex {
			mode: u32 = "a_Mode",
			pos: [f32; 2] = "a_Pos",
			uv: [f32; 2] = "a_Uv",
			color: [f32; 4] = "a_Color",
		}

		pipeline pipe {
			vbuf: gfx::VertexBuffer<Vertex> = (),
			color: gfx::TextureSampler<[f32; 4]> = "t_Color",
			out: gfx::BlendTarget<ColorFormat> = ("f_Color", ::gfx::state::MASK_ALL, ::gfx::preset::blend::ALPHA),
		}
	}
}
use self::gfx_impl::{ pipe, Vertex };

// Convenience constructor
impl Vertex {
	fn new(pos: [f32; 2], uv: [f32; 2], color: [f32; 4], mode: u32) -> Vertex {
		Vertex {
			pos: pos,
			uv: uv,
			color: color,
			mode: mode,
		}
	}
}

/// Possible errors that may occur during a call to `Renderer::new`.
#[derive(Debug)]
pub enum RendererCreationError {
    /// Errors that might occur when creating the glyph cache texture.
    Texture(),
    /// Errors that might occur when constructing the shader program.
    Program(gfx::PipelineStateError<String>),
	/// Errors that might occur when constructing the window.
	Window,
}

/// Possible errors that may occur during a call to `Renderer::fill`.
#[derive(Copy, Clone, Debug)]
pub enum FillError {
	/// Errors that might occur interacting with the window.
	Window,
}

/// Possible errors that may occur during a call to `Renderer::draw`.
#[derive(Debug)]
pub enum DrawError {
    /// Errors that might occur upon construction of a `glium::VertexBuffer`.
    Buffer(),
    /// Errors that might occur when drawing to the `glium::Surface`.
    Draw(),
	ContextError(glutin::ContextError),
}

pub struct GlyphCache {
	cache: text::GlyphCache,
	texture: gfx_core::handle::Texture<Resources, SurfaceFormat>,
	view: gfx_core::handle::ShaderResourceView<Resources, [f32; 4]>,
}

impl GlyphCache {
	pub fn new<F>(window: &glutin::Window, factory: &mut F) -> Result<Self, RendererCreationError>
		where F: gfx::Factory<Resources>
	{
		let (win_w, win_h) = match window.get_inner_size() {
			Some(s) => s,
			None => return Err(RendererCreationError::Window),
		};

		let dpi = window.hidpi_factor();
		let width = (win_w as f32 * dpi) as u32;
		let height = (win_h as f32 * dpi) as u32;

		const SCALE_TOLERANCE: f32 = 0.1;
		const POSITION_TOLERANCE: f32 = 0.1;

		let cache = text::GlyphCache::new(width, height,
			SCALE_TOLERANCE,
			POSITION_TOLERANCE);

		let data = vec![0; (width * height * 4) as usize];

		let (texture, view) = create_texture(factory, width, height, &data);

		Ok(Self {
			cache: cache,
			texture: texture,
			view: view,
		})
	}

	pub fn texture(&self) -> &gfx_core::handle::Texture<Resources, SurfaceFormat> {
		&self.texture
	}
}

pub struct Renderer {
	window: glutin::Window,
	glyph_cache: GlyphCache,
	encoder: gfx::Encoder<Resources, CommandBuffer>,
	vertices: Vec<Vertex>,
	device: Device,
	factory: Factory,
	data: pipe::Data<Resources>,
	pso: gfx::PipelineState<Resources, pipe::Meta>,
	main_color: gfx_core::handle::RenderTargetView<Resources, (gfx_core::format::R8_G8_B8_A8, gfx_core::format::Srgb)>,
}

impl Renderer {
	pub fn new(builder: glutin::WindowBuilder) -> Result<Self, RendererCreationError> {
		// Initialize gfx things
        let (window, mut device, mut factory, main_color, _) =
            gfx_window_glutin::init::<ColorFormat, DepthFormat>(builder);
        let mut encoder: gfx::Encoder<_, _> = factory.create_command_buffer().into();

		let (win_w, win_h) = match window.get_inner_size() {
			Some(s) => s,
			None => return Err(RendererCreationError::Window),
		};

        // Create texture sampler
        let sampler_info = texture::SamplerInfo::new(
            texture::FilterMethod::Bilinear,
            texture::WrapMode::Clamp
        );
        let sampler = {
			use gfx::Factory;
			factory.create_sampler(sampler_info)
		};

        // Dummy values for initialization
        let vbuf = factory.create_vertex_buffer(&[]);
        let (_, fake_texture) = create_texture(&mut factory, 2, 2, &[0; 4]);

        let mut data = pipe::Data {
            vbuf: vbuf,
            color: (fake_texture.clone(), sampler),
            out: main_color.clone(),
        };

        // Compile GL program
        let pso = factory.create_pipeline_simple(VERTEX_SHADER, FRAGMENT_SHADER, pipe::new())?;

		let glyph_cache = GlyphCache::new(&window, &mut factory)?;

		Ok(Self {
			encoder: encoder,
			glyph_cache: glyph_cache,
			window: window,
			vertices: Vec::new(),
			device: device,
			factory: factory,
			data: data,
			pso: pso,
			main_color: main_color,
		})
	}

	pub fn fill<P>(&mut self, mut primitives: P) -> Result<(), FillError> 
		where P: render::PrimitiveWalker
	{
		let Renderer { 
			ref mut encoder, 
			ref mut vertices, 
			ref mut glyph_cache,
			ref mut window,
			..
		} = *self;

		// If the window is closed, this will be None for one tick, so to avoid panicking with
		// unwrap, instead break the loop
		let (win_w, win_h) = match window.get_inner_size() {
			Some(s) => s,
			None => return Err(FillError::Window),
		};
		let dpi_factor = window.hidpi_factor();
		let (screen_width, screen_height) = (win_w as f32 * dpi_factor, win_h as f32 * dpi_factor);
		
		let half_win_w = win_w as Scalar / 2.0;
        let half_win_h = win_h as Scalar / 2.0;

		// Functions for converting for conrod scalar coords to GL vertex coords (-1.0 to 1.0).
        let vx = |x: Scalar| (x * dpi_factor as Scalar / half_win_w) as f32;
        let vy = |y: Scalar| (y * dpi_factor as Scalar / half_win_h) as f32;

        vertices.clear();

		// Create vertices
		while let Some(render::Primitive { id, kind, scizzor, rect }) = primitives.next_primitive() {
			match kind {
				render::PrimitiveKind::Rectangle { color } => {
					let color = color.to_fsa();
					let (l, r, b, t) = rect.l_r_b_t();

					let v = |x, y| {
                        Vertex {
                            pos: [vx(x), vy(y)],
                            uv: [0.0, 0.0],
                            color: color,
                            mode: MODE_GEOMETRY,
                        }
                    };

					let mut push_v = |x, y| vertices.push(v(x, y));

                    // Bottom left triangle.
                    push_v(l, t);
                    push_v(r, b);
                    push_v(l, b);

                    // Top right triangle.
                    push_v(l, t);
                    push_v(r, b);
                    push_v(r, t);
				},
				render::PrimitiveKind::Polygon { .. } => {
				},
				render::PrimitiveKind::Lines { .. } => {
				},
				render::PrimitiveKind::Image { .. } => {
				},
				render::PrimitiveKind::Text { color, text, font_id } => {
					let GlyphCache { ref mut cache, ref mut texture, .. } = *glyph_cache;

					let positioned_glyphs = text.positioned_glyphs(dpi_factor);

					// Queue the glyphs to be cached
					for glyph in positioned_glyphs {
						cache.queue_glyph(font_id.index(), glyph.clone());
					}

					cache.cache_queued(|rect, data| {
						let offset = [rect.min.x as u16, rect.min.y as u16];
						let size = [rect.width() as u16, rect.height() as u16];

						let new_data = data.iter().map(|x| [0, 0, 0, *x]).collect::<Vec<_>>();

						update_texture(encoder, texture, offset, size, &new_data);
					}).unwrap();

					let color = color.to_fsa();
					let cache_id = font_id.index();
					let origin = rt::point(0.0, 0.0);

					// A closure to convert RustType rects to GL rects
					let to_gl_rect = |screen_rect: rt::Rect<i32>| rt::Rect {
						min: origin
							+ (rt::vector(screen_rect.min.x as f32 / screen_width - 0.5,
											1.0 - screen_rect.min.y as f32 / screen_height - 0.5)) * 2.0,
						max: origin
							+ (rt::vector(screen_rect.max.x as f32 / screen_width - 0.5,
											1.0 - screen_rect.max.y as f32 / screen_height - 0.5)) * 2.0,
					};

					// Create new vertices
					let extension = positioned_glyphs.into_iter()
						.filter_map(|g| cache.rect_for(cache_id, g).ok().unwrap_or(None))
						.flat_map(|(uv_rect, screen_rect)| {
							use std::iter::once;

							let gl_rect = to_gl_rect(screen_rect);
							let v = |pos, uv| once(Vertex::new(pos, uv, color, MODE_TEXT));

							v([gl_rect.min.x, gl_rect.max.y], [uv_rect.min.x, uv_rect.max.y])
								.chain(v([gl_rect.min.x, gl_rect.min.y], [uv_rect.min.x, uv_rect.min.y]))
								.chain(v([gl_rect.max.x, gl_rect.min.y], [uv_rect.max.x, uv_rect.min.y]))
								.chain(v([gl_rect.max.x, gl_rect.min.y], [uv_rect.max.x, uv_rect.min.y]))
								.chain(v([gl_rect.max.x, gl_rect.max.y], [uv_rect.max.x, uv_rect.max.y]))
								.chain(v([gl_rect.min.x, gl_rect.max.y], [uv_rect.min.x, uv_rect.max.y]))
						});

					vertices.extend(extension);
				},
				render::PrimitiveKind::Other(_) => {},
			}
		}

		Ok(())
	}

	pub fn draw(&mut self) -> Result<(), DrawError> {
		use gfx_core::Device;

		let Renderer { 
			ref mut encoder,
			ref vertices,
			ref mut window,
			ref mut device,
			ref mut factory,
			ref mut data,
			ref pso,
			ref main_color,
			ref glyph_cache,
			..
		} = *self;

		let GlyphCache { ref view, .. } = *glyph_cache;

		// Clear the window
		encoder.clear(&main_color, [0.2, 0.2, 0.2, 1.0]);

		// Draw the vertices
		data.color.0 = view.clone();
		let (vbuf, slice) = factory.create_vertex_buffer_with_slice(vertices, ());
		data.vbuf = vbuf;
		encoder.draw(&slice, pso, data);

		// Display the results
		encoder.flush(device);
		window.swap_buffers()?;
		device.cleanup();

		Ok(())
	}

	pub fn window(&self) -> &glutin::Window {
		&self.window
	}
}

// Creates a gfx texture with the given data
fn create_texture<F, R>(factory: &mut F, width: u32, height: u32, data: &[u8])
	-> (gfx::handle::Texture<R, SurfaceFormat>, gfx::handle::ShaderResourceView<R, [f32; 4]>)

	where R: gfx::Resources, F: gfx::Factory<R>
{
	// Modified `Factory::create_texture_immutable_u8` for dynamic texture.
	fn create_texture<T, F, R>(
		factory: &mut F,
		kind: gfx::texture::Kind,
		data: &[&[u8]]
	) -> Result<(
		gfx::handle::Texture<R, T::Surface>,
		gfx::handle::ShaderResourceView<R, T::View>
	), gfx::CombinedError>
		where F: gfx::Factory<R>,
				R: gfx::Resources,
				T: gfx::format::TextureFormat
	{
		use gfx::{format, texture};
		use gfx::memory::{Usage, SHADER_RESOURCE};
		use gfx_core::memory::Typed;

		let surface = <T::Surface as format::SurfaceTyped>::get_surface_type();
		let num_slices = kind.get_num_slices().unwrap_or(1) as usize;
		let num_faces = if kind.is_cube() {6} else {1};
		let desc = texture::Info {
			kind: kind,
			levels: (data.len() / (num_slices * num_faces)) as texture::Level,
			format: surface,
			bind: SHADER_RESOURCE,
			usage: Usage::Dynamic,
		};
		let cty = <T::Channel as format::ChannelTyped>::get_channel_type();
		let raw = try!(factory.create_texture_raw(desc, Some(cty), Some(data)));
		let levels = (0, raw.get_info().levels - 1);
		let tex = Typed::new(raw);
		let view = try!(factory.view_texture_as_shader_resource::<T>(
			&tex, levels, format::Swizzle::new()
		));
		Ok((tex, view))
	}

	let kind = texture::Kind::D2(
		width as texture::Size,
		height as texture::Size,
		texture::AaMode::Single
	);
	create_texture::<ColorFormat, F, R>(factory, kind, &[data]).unwrap()
}

// Updates a texture with the given data (used for updating the GlyphCache texture)
fn update_texture<R, C>(encoder: &mut gfx::Encoder<R, C>,
						texture: &gfx::handle::Texture<R, SurfaceFormat>,
						offset: [u16; 2],
						size: [u16; 2],
						data: &[[u8; 4]])

	where R: gfx::Resources, C: gfx::CommandBuffer<R>
{
	let info = texture::ImageInfoCommon {
			xoffset: offset[0],
			yoffset: offset[1],
			zoffset: 0,
			width: size[0],
			height: size[1],
			depth: 0,
			format: (),
			mipmap: 0,
	};

	encoder.update_texture::<SurfaceFormat, FullFormat>(texture, None, info, data).unwrap();
}

impl From<gfx::PipelineStateError<String>> for RendererCreationError {
    fn from(err: gfx::PipelineStateError<String>) -> Self {
        RendererCreationError::Program(err)
    }
}

impl From<glutin::ContextError> for DrawError {
    fn from(err: glutin::ContextError) -> Self {
        DrawError::ContextError(err)
    }
}
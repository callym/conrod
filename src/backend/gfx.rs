//! A gfx-rs backend for rendering conrod primitives.

pub use gfx;
pub use glutin;

use gfx::{ CommandBuffer, Resources, texture };
use gfx::traits::FactoryExt;

use gfx_device_gl;

use { Rect, Scalar };
use color;
use image;
use render;
use std;
use text;
use text::rt;

/// A `Command` describing a step in the drawing process.
#[derive(Clone, Debug)]
pub enum Command<'a> {
    /// Draw to the target.
    Draw(Draw<'a>),
    /// Update the scizzor within the `glium::DrawParameters`.
    Scizzor(gfx::Rect),
}

/// A `Command` for drawing to the target.
///
/// Each variant describes how to draw the contents of the vertex buffer.
#[derive(Clone, Debug)]
pub enum Draw<'a> {
    /// A range of vertices representing triangles textured with the image in the
    /// image_map at the given `widget::Id`.
    Image(image::Id, &'a [Vertex]),
    /// A range of vertices representing plain triangles.
    Plain(&'a [Vertex]),
}

enum PreparedCommand {
    Image(image::Id, std::ops::Range<usize>),
    Plain(std::ops::Range<usize>),
    Scizzor(gfx::Rect),
}

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

/// Gfx textures that have two dimensions.
pub struct Texture<R: Resources> {
    texture: gfx::handle::Texture<R, SurfaceFormat>,
    view: gfx::handle::ShaderResourceView<R, [f32; 4]>,
}

impl<R: Resources> Texture<R> {
    pub fn new(texture: gfx::handle::Texture<R, SurfaceFormat>, view: gfx::handle::ShaderResourceView<R, [f32; 4]>) -> Self {
        Self {
            texture: texture,
            view: view,
        }
    }

    fn dimensions(&self) -> (u32, u32) {
        let d = self.texture.get_info().kind.get_dimensions();
		(d.0 as u32, d.1 as u32)
    }
}

/// Converts gamma (brightness) from sRGB to linear color space.
///
/// sRGB is the default color space for image editors, pictures, internet etc.
/// Linear gamma yields better results when doing math with colors.
pub fn gamma_srgb_to_linear(c: [f32; 4]) -> [f32; 4] {
    fn component(f: f32) -> f32 {
        // Taken from https://github.com/PistonDevelopers/graphics/src/color.rs#L42
        if f <= 0.04045 {
            f / 12.92
        } else {
            ((f + 0.055) / 1.055).powf(2.4)
        }
    }
    [component(c[0]), component(c[1]), component(c[2]), c[3]]
}

/// The format used for the Gfx color buffer
pub type ColorFormat = gfx::format::Srgba8;
/// The format used for the Gfx depth buffer
pub type DepthFormat = gfx::format::DepthStencil;
/// The format used for Gfx textures and surfaces
pub type SurfaceFormat = gfx::format::R8_G8_B8_A8;
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
            scissor: gfx::Scissor = (),
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
    Encoder,
}

/// Possible errors that may occur during a call to `Renderer::draw`.
#[derive(Debug)]
pub enum DrawError {
    /// Errors that might occur upon construction of a `glium::VertexBuffer`.
    Buffer(),
    /// Errors that might occur when drawing to the `glium::Surface`.
    Draw(),
    ContextError(glutin::ContextError),
    Encoder,
}

pub struct GlyphCache<R: Resources> {
    cache: text::GlyphCache,
    texture: gfx::handle::Texture<R, SurfaceFormat>,
    view: gfx::handle::ShaderResourceView<R, [f32; 4]>,
}

impl<R: Resources> GlyphCache<R> {
    pub fn new<F>(factory: &mut F, (width, height): (u16, u16), dpi_factor: f32) -> Result<Self, RendererCreationError>
        where F: Factory<R>
    {
        let width = (width as f32 * dpi_factor) as u32;
        let height = (width as f32 * dpi_factor) as u32;

        const SCALE_TOLERANCE: f32 = 0.1;
        const POSITION_TOLERANCE: f32 = 0.1;

        let cache = text::GlyphCache::new(
            width,
            height,
            SCALE_TOLERANCE,
            POSITION_TOLERANCE
        );

        let data = vec![0; (width * height * 4) as usize];

        let (texture, view) = create_texture(factory, width, height, &data);

        Ok(Self {
            cache: cache,
            texture: texture,
            view: view,
        })
    }
}

/// An iterator yielding `Command`s, produced by the `Renderer::commands` method.
pub struct Commands<'a> {
    commands: std::slice::Iter<'a, PreparedCommand>,
    vertices: &'a [Vertex],
}

impl<'a> Iterator for Commands<'a> {
    type Item = Command<'a>;
    fn next(&mut self) -> Option<Self::Item> {
        let Commands { ref mut commands, ref vertices } = *self;
        commands.next().map(|command| match *command {
            PreparedCommand::Scizzor(scizzor) => Command::Scizzor(scizzor),
            PreparedCommand::Plain(ref range) =>
                Command::Draw(Draw::Plain(&vertices[range.clone()])),
            PreparedCommand::Image(id, ref range) =>
                Command::Draw(Draw::Image(id, &vertices[range.clone()])),
        })
    }
}

/// A type used for translating `render::Primitives` into `Command`s that indicate how to draw the
/// conrod GUI using `gfx-rs`.
pub struct Renderer<R: Resources, C: CommandBuffer<R>> {
    glyph_cache: GlyphCache<R>,
    commands: Vec<PreparedCommand>,
    vertices: Vec<Vertex>,
    vertex_buffer: gfx::handle::Buffer<R, Vertex>,
    data: pipe::Data<R>,
    pso: gfx::PipelineState<R, pipe::Meta>,
    encoder: Option<gfx::Encoder<R, C>>,
}

impl<R: Resources, C: CommandBuffer<R>> Renderer<R, C> {
    /// Construct a new empty `Renderer`
    pub fn new<F>(factory: &mut F, main_color: &gfx::handle::RenderTargetView<R, (gfx::format::R8_G8_B8_A8, gfx::format::Srgb)>, (width, height): (u16, u16), dpi_factor: f32)
        -> Result<Self, RendererCreationError>
        where F: Factory<R, CommandBuffer = C>
    {
        // Create texture sampler
        let sampler_info = texture::SamplerInfo::new(
            texture::FilterMethod::Bilinear,
            texture::WrapMode::Clamp
        );
        let sampler = factory.create_sampler(sampler_info);

        // Dummy values for initialization
        let vbuf = factory.create_vertex_buffer(&[]);
        let (_, fake_texture) = create_texture(factory, 2, 2, &[0; 4]);

        let mut data = pipe::Data {
            vbuf: vbuf.clone(),
            color: (fake_texture.clone(), sampler),
            out: main_color.clone(),
            scissor: gfx::Rect {
                x: 0,
                y: 0,
                w: width,
                h: height,
            },
        };

        // Compile GL program
        let pso = {
            let set = factory.create_shader_set(
                VERTEX_SHADER,
                FRAGMENT_SHADER
            ).unwrap();
            factory.create_pipeline_state(
                &set,
                gfx::Primitive::TriangleList,
                gfx::state::Rasterizer {
                    samples: Some(gfx::state::MultiSample {}),
                    .. gfx::state::Rasterizer::new_fill()
                },
                pipe::new()
            )?
        };

        let glyph_cache = GlyphCache::new(factory, (width, height), dpi_factor)?;

        Ok(Self {
            glyph_cache: glyph_cache,
            commands: Vec::new(),
            vertices: Vec::new(),
            vertex_buffer: vbuf.clone(),
            
            data: data,
            pso: pso,
            encoder: None,
        })
    }

    /// You have to pass in a new Gfx::Encoder every time
    /// you want to draw, because at the end of the draw,
    /// we take the encoder out of the Renderer object to be used
    pub fn add_encoder(&mut self, encoder: gfx::Encoder<R, C>) {
        self.encoder = Some(encoder);
    }

    /// This function is checks if the current vertex buffer is big enough to 
    /// contain all the produced by the fill, and if it isn't big enough,
    /// it creates a new buffer that is big enough.
    /// 
    /// This means that the `draw` and `fill` function doesn't rely on a Gfx::Factory, which
    /// should mean that it's possible to thread them, as long as this is called
    /// in between the calls
    pub fn update_buffer<F>(&mut self, factory: &mut F) where F: Factory<R, CommandBuffer = C> {
        let Renderer { ref vertices, ref mut vertex_buffer, .. } = *self;
        
        if vertex_buffer.len() < vertices.len() {
            *vertex_buffer = factory.create_buffer(
                vertices.len(),
                gfx::buffer::Role::Vertex,
                gfx::memory::Usage::Dynamic,
                gfx::Bind::empty()
            ).unwrap();
        }
    }

    /// Fill the inner vertex and command buffers by translating the given `primitives`.
    /// Because you don't need to pass in a factory, this can (in theory) be threaded.
    pub fn fill<P>(&mut self, mut primitives: P, image_map: &image::Map<Texture<R>>, (width, height): (u16, u16), dpi_factor: f32) -> Result<(), FillError> 
        where P: render::PrimitiveWalker
    {
        self.commands.clear();
        self.vertices.clear();

        enum State {
            Image { image_id: image::Id, start: usize },
            Plain { start: usize },
        }

        let mut current_state = State::Plain { start: 0 };
        
        let (screen_width, screen_height) = (width as f32 * dpi_factor, height as f32 * dpi_factor);
        
        let half_win_w = width as f32 / 2.0;
        let half_win_h = height as f32 / 2.0;

        // Functions for converting for conrod scalar coords to GL vertex coords (-1.0 to 1.0).
        let vx = |x: Scalar| (x as f32 * dpi_factor / half_win_w) as f32;
        let vy = |y: Scalar| (y as f32 * dpi_factor / half_win_h) as f32;

        let mut current_scizzor = gfx::Rect {
            x: 0,
            w: screen_width as u16,
            y: 0,
            h: screen_height as u16,
        };

        let rect_to_gfx_rect = |rect: Rect| {
            let (w, h) = rect.w_h();
            let w = w as f32;
            let h = h as f32;
            let left = rect.left() as f32;
            let bottom = rect.bottom() as f32;

            let left = (left * dpi_factor + half_win_w) as u16;
            let bottom = (bottom * dpi_factor + half_win_h) as u16;
            let width = (w * dpi_factor) as u16;
            let height = (h * dpi_factor) as u16;

            gfx::Rect {
                x: std::cmp::max(left, 0),
                y: std::cmp::max(bottom, 0),
                w: std::cmp::min(width, screen_width as u16),
                h: std::cmp::min(height, screen_height as u16),
            }
        };

        // Create vertices
        while let Some(primitive) = primitives.next_primitive() {
            let Renderer { ref mut commands, ref mut encoder, ref mut vertices, .. } = *self;
            let encoder = match *encoder {
                Some(ref mut encoder) => encoder,
                None => return Err(FillError::Encoder)
            };

            // Switches to the `Plain` state and completes the previous `Command` if not already in the
            // `Plain` state.
            macro_rules! switch_to_plain_state {
                () => {
                    match current_state {
                        State::Plain { .. } => (),
                        State::Image { image_id, start } => {
                            commands.push(PreparedCommand::Image(image_id, start..vertices.len()));
                            current_state = State::Plain { start: vertices.len() };
                        },
                    }
                };
            }

            let render::Primitive { kind, scizzor, rect, .. } = primitive;

            // Check for a `Scizzor` command.
            let new_scizzor = rect_to_gfx_rect(scizzor);
            if new_scizzor != current_scizzor {
                // Finish the current command.
                match current_state {
                    State::Plain { start } =>
                        commands.push(PreparedCommand::Plain(start..vertices.len())),
                    State::Image { image_id, start } =>
                        commands.push(PreparedCommand::Image(image_id, start..vertices.len())),
                }

                // Update the scizzor and produce a command.
                current_scizzor = new_scizzor;
                commands.push(PreparedCommand::Scizzor(new_scizzor));

                // Set the state back to plain drawing.
                current_state = State::Plain { start: vertices.len() };
            }

            match kind {
                render::PrimitiveKind::Rectangle { color } => {
                    switch_to_plain_state!();

                    let color = gamma_srgb_to_linear(color.to_fsa());
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
                render::PrimitiveKind::Polygon { color, points } => {
                    // If we don't at least have a triangle, keep looping.
                    if points.len() < 3 {
                        continue;
                    }

                    switch_to_plain_state!();

                    let color = gamma_srgb_to_linear(color.to_fsa());

                    let v = |p: [Scalar; 2]| {
                        Vertex {
                            pos: [vx(p[0]), vy(p[1])],
                            uv: [0.0, 0.0],
                            color: color,
                            mode: MODE_GEOMETRY,
                        }
                    };

                    // Triangulate the polygon.
                    //
                    // Make triangles between the first point and every following pair of
                    // points.
                    //
                    // For example, for a polygon with 6 points (a to f), this makes the
                    // following triangles: abc, acd, ade, aef.
                    let first = points[0];
                    let first_v = v(first);
                    let mut prev_v = v(points[1]);
                    for &p in &points[2..] {
                        let v = v(p);
                        vertices.push(first_v);
                        vertices.push(prev_v);
                        vertices.push(v);
                        prev_v = v;
                    }
                },
                render::PrimitiveKind::Lines { color, thickness, points, .. } => {
                    // We need at least two points to draw any lines.
                    if points.len() < 2 {
                        continue;
                    }

                    switch_to_plain_state!();

                    let color = gamma_srgb_to_linear(color.to_fsa());

                    let v = |p: [Scalar; 2]| {
                        Vertex {
                            pos: [vx(p[0]), vy(p[1])],
                            uv: [0.0, 0.0],
                            color: color,
                            mode: MODE_GEOMETRY,
                        }
                    };

                    // Convert each line to a rectangle for triangulation.
                    //
                    // TODO: handle `cap` and properly join consecutive lines considering
                    // the miter. Discussion here:
                    // https://forum.libcinder.org/topic/smooth-thick-lines-using-geometry-shader#23286000001269127
                    let mut a = points[0];
                    for &b in &points[1..] {

                        let direction = [b[0] - a[0], b[1] - a[1]];
                        let mag = (direction[0].powi(2) + direction[1].powi(2)).sqrt();
                        let unit = [direction[0] / mag, direction[1] / mag];
                        let normal = [-unit[1], unit[0]];
                        let half_thickness = thickness / 2.0;

                        // A perpendicular line with length half the thickness.
                        let n = [normal[0] * half_thickness, normal[1] * half_thickness];

                        // The corners of the rectangle as GL vertices.
                        let (r1, r2, r3, r4);
                        r1 = v([a[0] + n[0], a[1] + n[1]]);
                        r2 = v([a[0] - n[0], a[1] - n[1]]);
                        r3 = v([b[0] + n[0], b[1] + n[1]]);
                        r4 = v([b[0] - n[0], b[1] - n[1]]);

                        // Push the rectangle's vertices.
                        let mut push_v = |v| vertices.push(v);
                        push_v(r1);
                        push_v(r4);
                        push_v(r2);
                        push_v(r1);
                        push_v(r4);
                        push_v(r3);

                        a = b;
                    }
				},
				render::PrimitiveKind::Image { image_id, color, source_rect } => {
					// Switch to the `Image` state for this image if we're not in it already.
                    let new_image_id = image_id;
                    match current_state {
                        // If we're already in the drawing mode for this image, we're done.
                        State::Image { image_id, .. } if image_id == new_image_id => (),
                        // If we were in the `Plain` drawing state, switch to Image drawing state.
                        State::Plain { start } => {
                            commands.push(PreparedCommand::Plain(start..vertices.len()));
                            current_state = State::Image {
                                image_id: new_image_id,
                                start: vertices.len(),
                            };
                        },
                        // If we were drawing a different image, switch state to draw *this* image.
                        State::Image { image_id, start } => {
                            commands.push(PreparedCommand::Image(image_id, start..vertices.len()));
                            current_state = State::Image {
                                image_id: new_image_id,
                                start: vertices.len(),
                            };
                        },
                    }

                    let color = color.unwrap_or(color::WHITE).to_fsa();

                    let (image_w, image_h) = image_map.get(&image_id).unwrap().dimensions();
                    let (image_w, image_h) = (image_w as Scalar, image_h as Scalar);

                    // Get the sides of the source rectangle as uv coordinates.
                    //
                    // Texture coordinates range:
                    // - left to right: 0.0 to 1.0
                    // - bottom to top: 0.0 to 1.0
                    let (uv_l, uv_r, uv_b, uv_t) = match source_rect {
                        Some(src_rect) => {
                            let (l, r, b, t) = src_rect.l_r_b_t();
                            ((l / image_w) as f32,
                             (r / image_w) as f32,
                             (b / image_h) as f32,
                             (t / image_h) as f32)
                        },
                        None => (0.0, 1.0, 0.0, 1.0),
                    };

                    let v = |x, y, t| {
                        Vertex {
                            pos: [vx(x), vy(y)],
                            uv: t,
                            color: color,
                            mode: MODE_IMAGE,
                        }
                    };

                    let mut push_v = |x, y, t| vertices.push(v(x, y, t));

                    let (l, r, b, t) = rect.l_r_b_t();

                    // Bottom left triangle.
                    push_v(l, t, [uv_l, uv_b]);
                    push_v(r, b, [uv_r, uv_t]);
                    push_v(l, b, [uv_l, uv_t]);

                    // Top right triangle.
                    push_v(l, t, [uv_l, uv_b]);
                    push_v(r, b, [uv_r, uv_t]);
                    push_v(r, t, [uv_r, uv_b]);
				},
				render::PrimitiveKind::Text { color, text, font_id } => {
                    switch_to_plain_state!();

                    let positioned_glyphs = text.positioned_glyphs(dpi_factor);

                    let GlyphCache { ref mut cache, ref mut texture, .. } = self.glyph_cache;

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

                    let color = gamma_srgb_to_linear(color.to_fsa());
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

        // Enter the final command.
        match current_state {
            State::Plain { start } =>
                self.commands.push(PreparedCommand::Plain(start..self.vertices.len())),
            State::Image { image_id, start } =>
                self.commands.push(PreparedCommand::Image(image_id, start..self.vertices.len())),
        }

        Ok(())
    }

    /// Fills a Gfx::Encoder with commands to be drawn onto a display
    ///
    /// Note: This does not draw the commands because (in theory), you could
    /// call this method from another thread and then draw to the display in
    /// your main thread using the produced Encoder.
    pub fn draw(&mut self, image_map: &image::Map<Texture<R>>) -> Result<gfx::Encoder<R, C>, DrawError>
    {
        // needs to indent this block so the references are
        // dropped before we try to end the frame
        {
            let Renderer { 
                ref glyph_cache,
                ref commands,
                ref vertices,
                ref mut data,
                ref pso,
                ref mut encoder,
                ref mut vertex_buffer,
                ..
            } = *self;

            let encoder = match *encoder {
                Some(ref mut encoder) => encoder,
                None => return Err(DrawError::Encoder)
            };

            let GlyphCache { ref view, .. } = *glyph_cache;

            let commands = Commands {
                commands: commands.iter(),
                vertices: vertices,
            };

            encoder.update_buffer(vertex_buffer, vertices, 0).unwrap();
            data.vbuf = vertex_buffer.clone();

            let mut start = 0;
            for command in commands {
                match command {
                    // Update the `scizzor` before continuing to draw.
                    Command::Scizzor(scizzor) => data.scissor = scizzor,
                    // Draw to the target with the given `draw` command.
                    Command::Draw(draw) => match draw {
                        // Draw text and plain 2D geometry.
                        Draw::Plain(slice) => {
                            // Draw the vertices
                            data.color.0 = view.clone();
                            let len = slice.len() as u32;
                            let slice = gfx::Slice {
                                start: start,
                                end: start + len,
                                base_vertex: 0,
                                instances: None,
                                buffer: gfx::IndexBuffer::Auto,
                            };
                            start += len;
                            encoder.draw(&slice, pso, data);
                        },

                        // Draw an image whose texture data lies within the `image_map` at the
                        // given `id`.
                        Draw::Image(image_id, slice) => {
                            // Draw the vertices
                            data.color.0 = image_map.get(&image_id).unwrap().view.clone();
                            let len = slice.len() as u32;
                            let slice = gfx::Slice {
                                start: start,
                                end: start + len,
                                base_vertex: 0,
                                instances: None,
                                buffer: gfx::IndexBuffer::Auto,
                            };
                            start += len;
                            encoder.draw(&slice, pso, data);
                        },

                    }
                }
            }
        }

        Ok(self.encoder.take().unwrap())
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

/// This trait is used as a cross-platform way
/// to create encoders for use in the renderer
pub trait Factory<R: Resources> : gfx::Factory<R> {
    /// The type of the CommandBuffer produced by this Gfx factory
    /// This is different for every Gfx backend (OGL, DX, Vulkan, ..)
    type CommandBuffer: CommandBuffer<R>;
    fn create_encoder(&mut self) -> gfx::Encoder<R, Self::CommandBuffer>;
}

impl Factory<gfx_device_gl::Resources> for gfx_device_gl::Factory {
    type CommandBuffer = gfx_device_gl::CommandBuffer;
    fn create_encoder(&mut self) -> gfx::Encoder<gfx_device_gl::Resources, Self::CommandBuffer> {
        self.create_command_buffer().into()
    }
}
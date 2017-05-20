//! A demonstration of using `winit` to provide events and GFX to draw the UI.
//!
//! `winit` is used via the `glutin` crate which also provides an OpenGL context for drawing
//! `conrod::render::Primitives` to the screen.

#![allow(unused_variables)]

#[cfg(all(feature="winit", feature="gfx-rs"))] #[macro_use] extern crate conrod;

#[cfg(all(feature="winit", feature="gfx-rs"))] mod support;

fn main() {
    feature::main();
}

#[cfg(all(feature="winit", feature="gfx-rs"))]
mod feature {
    extern crate find_folder;
    extern crate image;
    extern crate gfx_window_glutin;

    use conrod;
    use conrod::backend::gfx;
    use support;
    use std;

    const WIN_W: u16 = support::WIN_W as u16;
    const WIN_H: u16 = support::WIN_H as u16;

    pub fn main() {
        // Builder for window
        let builder = gfx::glutin::WindowBuilder::new()
            .with_vsync()
            .with_dimensions(WIN_W as u32, WIN_H as u32)
            .with_title("Conrod with GFX and Glutin")
            .with_multisampling(8);
        // Initialize gfx things
        let (window, mut device, mut factory, main_color, _) =
            gfx_window_glutin::init::<gfx::ColorFormat, gfx::DepthFormat>(builder);
        let mut encoder: gfx::gfx::Encoder<_, _> = factory.create_command_buffer().into();

        // Create Ui and Ids of widgets to instantiate
        let mut ui = conrod::UiBuilder::new([WIN_W as f64, WIN_H as f64]).theme(support::theme()).build();
        let ids = support::Ids::new(ui.widget_id_generator());

        // Load font from file
        let assets = find_folder::Search::KidsThenParents(3, 5).for_folder("assets").unwrap();
        let font_path = assets.join("fonts/NotoSans/NotoSans-Regular.ttf");
        ui.fonts.insert_from_file(font_path).unwrap();

        let mut renderer = conrod::backend::gfx::Renderer::new( 
            &mut factory, 
            &main_color, 
            (WIN_W, WIN_H), 
            window.hidpi_factor() 
        ).unwrap(); 

        // Load the Rust logo from our assets folder to use as an example image.
        fn load_rust_logo<R, C, F>(factory: &mut F) -> gfx::Texture<R>
            where R: gfx::gfx::Resources, C: gfx::gfx::CommandBuffer<R>, F: gfx::Factory<R, CommandBuffer = C>
        {
            use self::gfx::gfx::Factory;

            let assets = find_folder::Search::ParentsThenKids(3, 3).for_folder("assets").unwrap();
            let path = assets.join("images/rust.png");
            let rgba_image = image::open(&std::path::Path::new(&path)).unwrap().to_rgba();
            let (width, height) = rgba_image.dimensions();
            let kind = gfx::gfx::texture::Kind::D2(width as u16, height as u16, gfx::gfx::texture::AaMode::Single);
            let texture_view = factory.create_texture_immutable_u8::<gfx::ColorFormat>(kind, &[&rgba_image]).unwrap();

            gfx::Texture::new(texture_view.0, texture_view.1)
        }

        let mut image_map = conrod::image::Map::new();
        let rust_logo = image_map.insert(load_rust_logo(&mut factory));

        // Demonstration app state that we'll control with our conrod GUI.
        let mut app = support::DemoApp::new(rust_logo);

        let mut renderer = conrod::backend::gfx::Renderer::new(
            &mut factory,
            &main_color,
            (WIN_W, WIN_H),
            window.hidpi_factor()
        ).unwrap();

        // Event loop
        let mut event_loop = support::EventLoop::new();
        'main: loop {
            
            // Handle all events.
            for event in event_loop.next(&window) {

                // Use the `winit` backend feature to convert the winit event to a conrod one.
                if let Some(event) = conrod::backend::winit::convert(event.clone(), &window) {
                    ui.handle_event(event);
                    event_loop.needs_update();
                }

                match event {
                    // Break from the loop upon `Escape`.
                    gfx::glutin::Event::KeyboardInput(_, _, Some(gfx::glutin::VirtualKeyCode::Escape)) |
                    gfx::glutin::Event::Closed =>
                        break 'main,
                    _ => {},
                }
            }

            // Instantiate a GUI demonstrating every widget type provided by conrod.
            support::gui(&mut ui.set_widgets(), &ids, &mut app);

            // Draw the `Ui`.
            if let Some(primitives) = ui.draw_if_changed() {
                use self::gfx::gfx::Device;

                // Clear the window
                encoder.clear(&main_color, [0.2, 0.2, 0.2, 1.0]);
                encoder.flush(&mut device);

                renderer.fill(&mut factory, primitives, &image_map, (WIN_W, WIN_H), window.hidpi_factor());

                let mut encoder = renderer.draw(&image_map).unwrap();
                encoder.flush(&mut device);

                window.swap_buffers().unwrap();
                device.cleanup();
            }
        }
    }
}

#[cfg(not(all(feature="winit", feature="gfx-rs")))]
mod feature {
    pub fn main() {
        println!("This example requires the `winit` and `gfx-rs` features. \
                 Try running `cargo run --release --features=\"winit gfx-rs\" --example <example_name>`");
    }
}

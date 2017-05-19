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

    use conrod;
    use conrod::backend::gfx;
    use support;
    use std;

    const WIN_W: u32 = support::WIN_W;
    const WIN_H: u32 = support::WIN_H;

    pub fn main() {
        // Builder for window
        let builder = gfx::glutin::WindowBuilder::new()
            .with_title("Conrod with GFX and Glutin")
            .with_dimensions(WIN_W, WIN_H);

        // Create Ui and Ids of widgets to instantiate
        let mut ui = conrod::UiBuilder::new([WIN_W as f64, WIN_H as f64]).theme(support::theme()).build();
        let ids = support::Ids::new(ui.widget_id_generator());

        // Load font from file
        let assets = find_folder::Search::KidsThenParents(3, 5).for_folder("assets").unwrap();
        let font_path = assets.join("fonts/NotoSans/NotoSans-Regular.ttf");
        ui.fonts.insert_from_file(font_path).unwrap();

        let mut renderer = conrod::backend::gfx::Renderer::new(builder).unwrap();

        // Load the Rust logo from our assets folder to use as an example image.
        fn load_rust_logo(factory: &gfx::Factory) -> (
            gfx::gfx::handle::Texture<gfx::Resources, gfx::gfx::format::R8_G8_B8_A8>,
            gfx::gfx::handle::ShaderResourceView<gfx::Resources, [f32; 4]>
        )
        {
            use self::gfx::gfx::Factory;

            let assets = find_folder::Search::ParentsThenKids(3, 3).for_folder("assets").unwrap();
            let path = assets.join("images/rust.png");
            let rgba_image = image::open(&std::path::Path::new(&path)).unwrap().to_rgba();
            let (width, height) = rgba_image.dimensions();
            let kind = gfx::gfx::texture::Kind::D2(width as u16, height as u16, gfx::gfx::texture::AaMode::Single);
            let texture_view = factory.create_texture_immutable_u8::<gfx::gfx::format::Rgba8>(kind, &[&rgba_image]).unwrap();
            texture_view
        }

        let mut image_map = conrod::image::Map::new();
        let rust_logo = image_map.insert(load_rust_logo(renderer.factory()));

        // Demonstration app state that we'll control with our conrod GUI.
        let mut app = support::DemoApp::new(rust_logo);

        // Event loop
        let mut event_loop = support::EventLoop::new();
        'main: loop {
            
            // Handle all events.
            for event in event_loop.next(renderer.window()) {

                // Use the `winit` backend feature to convert the winit event to a conrod one.
                if let Some(event) = conrod::backend::winit::convert(event.clone(), renderer.window()) {
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
                renderer.fill(primitives, &image_map);

                renderer.draw();
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

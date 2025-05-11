#include <fstream>
#include <memory>
#include <array>
#include <future>
#include <iostream>

#define CL_HPP_TARGET_OPENCL_VERSION 200
#define CL_HPP_ENABLE_EXCEPTIONS
#include <CL/opencl.hpp>

namespace ray
{
#include "raylib.h"
}

const int scale_factor = 8;

const int table_w = 800/scale_factor;
const int table_h = 600/scale_factor;

struct OpenCLState
{
  std::vector<cl::Platform> platforms;
  std::vector<std::string> platformNames;
  std::vector<std::vector<cl::Device>> devices;
  std::vector<std::vector<std::string>> deviceNames;

  std::unique_ptr<cl::Context> context;
  std::unique_ptr<cl::CommandQueue> queue;
  std::unique_ptr<cl::Program> program;
  std::unique_ptr<cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int>> kernel;

  std::array<std::unique_ptr<cl::Buffer>, 2> buffers;
  int idx = 0;
  std::unique_ptr<cl::Buffer> staging_write;
  std::unique_ptr<cl::Buffer> staging_read;

  std::vector<unsigned int> table = std::vector<unsigned int>(table_w * table_h, 0);
  ray::Image image;
  ray::Texture2D texture;

  int iPlatform = -1;
  int iDevice = -1;
  bool init()
  {
    bool status = true;
    try
    {
      cl::Platform::get(&platforms);
      devices.resize(platforms.size());
      platformNames.resize(platforms.size());
      deviceNames.resize(platforms.size());
      int i = 0;
      for(auto& p : platforms)
      {
        platformNames[i] = p.getInfo<CL_PLATFORM_NAME>();
        p.getDevices(CL_DEVICE_TYPE_ALL, &devices[i]);
        deviceNames[i].resize(devices[i].size());
        int j = 0;
        for(auto& d : devices[i])
        {
          deviceNames[i][j] = d.getInfo<CL_DEVICE_NAME>();
          j += 1;
        }
        i += 1;
      }

      image = ray::GenImageColor(table_w, table_h, ray::RAYWHITE);
      texture = ray::LoadTextureFromImage(image);
    }
    catch(...)
    {
      status = false;
    }
    return status;
  }

  bool select(int iPlatform_in, int iDevice_in)
  {
    bool status = true;
    
    try
    {
      std::vector<cl_context_properties>cps{ CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[iPlatform_in]()), 0};
      context = std::make_unique<cl::Context>(devices[iPlatform_in][iDevice_in], cps.data());

      cl::QueueProperties qps{cl::QueueProperties::Profiling};
      queue = std::make_unique<cl::CommandQueue>(*context.get(), devices[iPlatform_in][iDevice_in], qps);

      // Load and compile kernel program:
      std::ifstream source{"./step.cl"};
      if( !source.is_open() ){ throw std::runtime_error{ std::string{"Error opening kernel file: step.cl"} }; }
      std::string source_string{ std::istreambuf_iterator<char>{ source },
                                 std::istreambuf_iterator<char>{} };
      program = std::make_unique<cl::Program>(*context.get(), source_string);
      program->build({devices[iPlatform_in][iDevice_in]});
      
      kernel = std::make_unique<cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer, int, int>>(*program.get(), "step");

      {
        auto set = [&](int row, int col){ table[row*table_w+col] = 1; };

        // Glider gun:
        // Left box:
        set(6, 2); set(6, 3);
        set(7, 2); set(7, 3);
        // Right box:
        set(4, 36); set(4, 37);
        set(5, 36); set(5, 37);
        // C-shape + arrow:
        set(4, 14); set(4, 15);
        set(5, 13); set(5, 17);
        set(6, 12); set(6, 18);
        set(7, 12); set(7, 18); set(7, 16); set(7, 19);
        set(8, 12); set(8, 18);
        set(9, 13); set(9, 17);
        set(10, 14); set(10, 15);
        // V-shape:
        set(2, 26);
        set(3, 24); set(3, 26);
        set(4, 22); set(4, 23);
        set(5, 22); set(5, 23);
        set(6, 22); set(6, 23);
        set(7, 24); set(7, 26);
        set(8, 26);
      }
    
      // Glider:
      /*table[16*table_w+16] = 1;
      table[17*table_w+17] = 1;
      table[17*table_w+18] = 1;
      table[18*table_w+16] = 1;
      table[18*table_w+17] = 1;*/

      buffers[0] = std::make_unique<cl::Buffer>(*context.get(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, table_w*table_h*sizeof(unsigned int) );
      buffers[1] = std::make_unique<cl::Buffer>(*context.get(), CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS, table_w*table_h*sizeof(unsigned int) );
      staging_write = std::make_unique<cl::Buffer>(*context.get(), CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, table_w*table_h*sizeof(unsigned int), table.data() );
      staging_read  = std::make_unique<cl::Buffer>(*context.get(), CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, table_w*table_h*sizeof(cl_char4) );

      queue->enqueueCopyBuffer(*staging_write.get(), *buffers[0].get(), 0, 0, table_w*table_h*sizeof(unsigned int));
      queue->finish();
      idx = 0;
    }
    catch(...)
    {
      status = false;
      kernel.reset();
      program.reset();
      queue.reset();
      context.reset();
    }
    if(status)
    {
      iPlatform = iPlatform_in;
      iDevice = iDevice_in;
    }
    return status;
  }

  void step()
  {
    try
    {
      // Launch kernel:
      cl::NDRange thread_count = {table_w, table_h};

      std::vector<cl::Event> event(1);
      event[0] = (*kernel.get())(cl::EnqueueArgs{*queue.get(), thread_count}, *staging_read.get(), *buffers[1-idx].get(), *buffers[idx].get(), table_w, table_h);
      
      // Copy back results:
      cl::Event event2;
      // Explicit copy:
      //queue->enqueueReadBuffer(*staging_read.get(), false, 0, table_w*table_h*sizeof(unsigned int), image.data, &event, &event2);

      // Mapping the buffer to host memory:
      {
        auto ptr = queue->enqueueMapBuffer(*staging_read.get(), true, CL_MAP_READ, 0, table_w*table_h*sizeof(unsigned int), &event);
        ray::UpdateTexture(texture, ptr);
        queue->enqueueUnmapMemObject(*staging_read.get(), ptr, nullptr, &event2);
      }

      // This should be also working:
      // queue->enqueueMigrateMemObjects({*staging_read.get()}, CL_MIGRATE_MEM_OBJECT_HOST, &event, &event2);
      
      // Synchronize:
      event2.wait();

      // Double buffering swap:
      idx = 1 - idx;
    }
    catch(...)
    {
      std::cout << "Error in step!\n";
    }
  }
};

int main()
{
  ray::SetConfigFlags(ray::FLAG_WINDOW_RESIZABLE); // Window configuration flags
  ray::InitWindow(2*16+table_w*scale_factor, 2*16+32*2+table_h*scale_factor, "Game of Life");
  ray::SetTargetFPS(120);

  OpenCLState cl;
  bool cl_state = cl.init();
  bool no_device_selected = true;
  bool switching = false;
  bool running = false;
  bool failed = false;
  bool once = false;

  std::future<bool> switcher;

  auto t0 = std::chrono::steady_clock::now();
  auto t1 = std::chrono::steady_clock::now();

  bool step_limiter = true;
  ray::Rectangle LimitClickArea{2*16+table_w/2*scale_factor, 16, 16, 24+4};
  std::string LimitClickText = "Toggle Limiter";
  LimitClickArea.width = ray::MeasureText(LimitClickText.c_str(), 24) + 4;

  while (!ray::WindowShouldClose())
  {
    


    // Stepping logic:
    if(running)
    {
      t1 = std::chrono::steady_clock::now();
      auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();
      if(once && (!step_limiter || dt > 50)){ once = false; }
      if(!once)
      {
        cl.step();
        once = true;
        t0 = std::chrono::steady_clock::now();
      }
    }

    // Rendering:
    ray::BeginDrawing();
    ray::ClearBackground(ray::BLACK);
    ray::DrawFPS(2*16+table_w*scale_factor-96, 4);

    if(running)
    {
      ray::DrawRectangleLinesEx(LimitClickArea, 2.0f, step_limiter ? ray::YELLOW : ray::BROWN);
      ray::DrawText(LimitClickText.c_str(), LimitClickArea.x+2, LimitClickArea.y+2, 24, ray::ORANGE);
      // Toggle limiter on click:
      if( ray::IsMouseButtonReleased(ray::MOUSE_BUTTON_LEFT) && ray::CheckCollisionPointRec(ray::GetMousePosition(), LimitClickArea))
      {
        step_limiter = !step_limiter;
      }
    }

    if(cl_state)
    {
      if(no_device_selected)
      {
        ray::DrawText("OpenCL Initialized", 16, 16, 24, ray::WHITE);
        ray::DrawText(("Found " + std::to_string(cl.platforms.size()) + " platforms:    (click on a device to select it)").c_str(), 16, 16+32, 24, ray::WHITE);
        int k = 0;
        for(int p=0; p<cl.platformNames.size(); ++p)
        {
          auto& s = cl.platformNames[p];
          ray::DrawText(s.c_str(), 16, 16+32*(2+k), 24, ray::WHITE);
          for(int l=0; l<cl.deviceNames[k].size(); ++l)
          {
            int width = ray::MeasureText(cl.deviceNames[k][l].c_str(), 24);
            ray::Rectangle area{16+386+64*l-2.0f, 16+32*(2+k)-2.0f, width+4.0f, 24+4.0f};
            ray::DrawRectangleLinesEx(area, 2.0f, k % 2 == 1 ? ray::LIGHTGRAY : ray::SKYBLUE);

            if( ray::IsMouseButtonReleased(ray::MOUSE_BUTTON_LEFT) && CheckCollisionPointRec(ray::GetMousePosition(), area))
            {
              switcher = std::async(std::launch::async, [&, p, l]{ return cl.select(p, l); });
              no_device_selected = false; switching = true;
            }
            ray::DrawText(cl.deviceNames[k][l].c_str(), 16+386+64*l, 16+32*(2+k), 24, ray::WHITE);
          }
          k += 1;
        }
      }
      else if(!no_device_selected && switching)
      {
        ray::DrawText("Switching to another OpenCL platform / device, please wait...", 16, 16, 24, ray::WHITE);
        if(!failed && !running && switcher.wait_for(std::chrono::microseconds{0}) == std::future_status::ready)
        {
          if(switcher.get())
          {
            switching = false;
            running = true;
          }
          else
          {
            switching = false;
            failed = true;
          }
        }
      }
      else if(running)
      {
        ray::DrawText(cl.platformNames[cl.iPlatform].c_str(), 16, 16, 24, ray::BLUE);
        ray::DrawText(cl.deviceNames[cl.iPlatform][cl.iDevice].c_str(), 16, 16+32, 24, ray::BLUE);
        // ray::UpdateTexture(cl.texture, cl.image.data);
        ray::DrawTextureEx(cl.texture, {16, 16+32*2}, 0, scale_factor, ray::WHITE);
      }
      else if(failed)
      {
        ray::DrawText("Device selection or Kernel compilation failed...", 16, 16, 24, ray::RED);  
      }
    }
    else
    {
      ray::DrawText("OpenCL initialization failed", 16, 16, 24, ray::RED);
    }
    ray::EndDrawing();
  }

  ray::CloseWindow();

  return 0;
}
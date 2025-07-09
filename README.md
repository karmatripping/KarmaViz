# 🌈 KarmaViz - Advanced Audio Visualizer

<div align="center">

![KarmaViz Logo](https://img.shields.io/badge/KarmaViz-Audio%20Visualizer-purple?style=for-the-badge&logo=music&logoColor=white)

[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)](https://python.org)
[![OpenGL](https://img.shields.io/badge/OpenGL-ModernGL-green?style=flat-square&logo=opengl)](https://moderngl.readthedocs.io/)
[![License](https://img.shields.io/badge/License-Personal%20Use-orange?style=flat-square)](LICENSE.md)

**A cutting-edge, GPU-accelerated audio visualizer for Linux with real-time GLSL shader compilation, advanced waveform rendering, and immersive visual effects.**

[🎵 Features](#-features) • [🚀 Installation](#-installation) • [⌨️ Controls](#️-keyboard-shortcuts) • [☕ Support](#-support-the-project)

</div>

---

## 🎵 Features

### 🎨 **Advanced Visual Effects**
- **GPU-Accelerated Rendering**: Leverages ModernGL for high-performance OpenGL rendering
- **Real-time GLSL Shader Compilation**: Dynamic shader loading with live error reporting
- **Customizable Waveform Rendering**: 100+ built-in waveforms across 12 categories
- **Dynamic Warp Maps**: 3D transformations, distortions, and geometric effects
- **Multi-layered Effects**: Glow, trails, smoke, pulse, bounce, and kaleidoscope effects
- **Advanced Color Palettes**: 30+ carefully crafted color schemes with smooth transitions
- **Spectrogram Overlay**: Spectral analysis for an added visual element**

### 🎵 **Audio Processing**
- **Real-time Audio Analysis**: Advanced FFT processing with beat detection
- **Multiple Audio Sources**: System audio capture with device selection
- **Adaptive Sensitivity**: Dynamic beat detection with configurable thresholds
- **Audio-reactive Parameters**: All visual effects respond to audio characteristics

### 🛠️ **Professional Tools**
- **Live Shader Editor**: Built-in GLSL editor with syntax highlighting and error reporting
- **Waveform Editor**: Create and modify custom waveforms with live preview
- **Warp Map Editor**: Design complex 3D transformations and distortion effects with live preview
- **Preset System**: Save and load complete visualization configurations
- **Performance Monitoring**: Real-time FPS and performance statistics

### 🎮 **User Experience**
- **Intuitive Controls**: Comprehensive keyboard shortcuts for most features
- **GUI Configuration**: Modern Qt-based settings interface
- **Fullscreen Support**: Multiple monitor support with resolution selection
- **Anti-aliasing**: FXAA post-processing for smooth visuals
- **Mouse Interaction**: Optional mouse-reactive effects

---

## 🚀 Installation

### Prerequisites
- **Linux Operating System** (Ubuntu/Debian/Arch/Fedora recommended)
- **Python 3.8+** (3.9+ recommended)
- **OpenGL 3.3+** compatible graphics card
- **Audio system** (ALSA/PulseAudio/JACK)
- **Git** for cloning the repository

### Method 1: Virtual Environment Setup (Recommended)

**Step 1: Clone and Setup Virtual Environment**
```bash
# Clone the repository
git clone https://github.com/KarmaTripping/karmaviz.git
cd karmaviz

# Create virtual environment
python3 -m venv karmaviz-env

# Activate virtual environment
source karmaviz-env/bin/activate
```
**Step 2: Install Dependencies**
```bash
# Upgrade pip to latest version
pip install --upgrade pip

# Install Python dependencies
pip install -r requirements.txt

# Install Cython for performance optimizations
pip install Cython
```
**Step 3: Build Cython Extensions**
```bash
# Build optimized Cython extensions for better performance
python setup.py build_ext --inplace

# Verify Cython compilation
python -c "import modules.color_ops; print('Cython extensions loaded successfully!')"
```
**Step 4: Run KarmaViz**
```bash
# Launch KarmaViz
python main.py

# For debug mode with verbose logging
python main.py --debug
```
### Method 2: Direct pip Installation
```bash
# Clone the repository
git clone https://github.com/karmatripping/karmaviz
cd karmaviz

# Install in development mode (recommended for contributors)
pip install -e .

# OR install normally
pip install .

# Build Cython extensions
python setup.py build_ext --inplace

# Run KarmaViz
python main.py
```
### Method 3: Quick Install (No Virtual Environment)
```bash
# Clone the repository
git clone https://github.com/KarmaTripping/karmaviz.git
cd karmaviz

# Install dependencies directly
pip install -r requirements.txt

# Build Cython extensions (optional, for better performance)
python setup.py build_ext --inplace

# Run KarmaViz
python main.py
```
### Dependencies

**Core Requirements:**
- `pygame` - Window management and input handling
- `moderngl` - OpenGL rendering
- `numpy` - Numerical computations
- `PyQt5` - GUI interface
- `sounddevice` - Audio capture
- `scipy` - Audio processing

**Optional (for better performance):**
- `Cython` - Compiled color operations

### Installation Troubleshooting

**Cython compilation fails:**
```bash
# Install build tools
sudo apt install build-essential python3-dev  # Ubuntu/Debian
sudo pacman -S base-devel python-devel        # Arch Linux
sudo dnf install gcc gcc-c++ python3-devel   # Fedora

# Reinstall Cython
pip uninstall Cython
pip install Cython
python setup.py build_ext --inplace
```
**Audio dependencies missing:**
```bash
# Ubuntu/Debian
sudo apt install portaudio19-dev libasound2-dev pulseaudio-dev

# Arch Linux
sudo pacman -S portaudio alsa-lib pulseaudio

# Fedora
sudo dnf install portaudio-devel alsa-lib-devel pulseaudio-libs-devel
```
**Qt5 installation issues:**
```bash
# If PyQt5 installation fails, try system package
sudo apt install python3-pyqt5 python3-pyqt5.qtopengl  # Ubuntu/Debian
sudo pacman -S python-pyqt5                             # Arch Linux
sudo dnf install python3-qt5                            # Fedora

# Or install via pip with specific version
pip install PyQt5==5.15.7
```
**Virtual environment activation:**
```bash
# If activation fails, ensure venv module is installed
sudo apt install python3-venv  # Ubuntu/Debian

# Create virtual environment with specific Python version
python3.9 -m venv karmaviz-env  # Use specific Python version
```
### System-Specific Setup

**Ubuntu/Debian:**
```bash
# Update package list
sudo apt update

# Install system dependencies
sudo apt install python3-dev python3-pip python3-venv
sudo apt install portaudio19-dev libasound2-dev
sudo apt install libgl1-mesa-dev libglu1-mesa-dev
sudo apt install build-essential

# For Qt5 GUI support
sudo apt install python3-pyqt5 python3-pyqt5.qtopengl

# Clone and setup KarmaViz
git clone https://github.com/karmatripping/karmaviz
cd karmaviz
python3 -m venv karmaviz-env
source karmaviz-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python setup.py build_ext --inplace
```
**Arch Linux:**
```bash
# Install system dependencies
sudo pacman -S python python-pip python-virtualenv
sudo pacman -S portaudio mesa base-devel
sudo pacman -S python-pyqt5

# Clone and setup KarmaViz
git clone https://github.com/KarmaTripping/karmaviz.git
cd karmaviz
python -m venv karmaviz-env
source karmaviz-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python setup.py build_ext --inplace
```
**Fedora/RHEL/CentOS:**
```bash
# Install system dependencies
sudo dnf install python3-devel python3-pip python3-virtualenv
sudo dnf install portaudio-devel alsa-lib-devel
sudo dnf install mesa-libGL-devel mesa-libGLU-devel
sudo dnf install gcc gcc-c++ make

# For Qt5 GUI support
sudo dnf install python3-qt5

# Clone and setup KarmaViz
git clone https://github.com/KarmaTripping/karmaviz.git
cd karmaviz
python3 -m venv karmaviz-env
source karmaviz-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python setup.py build_ext --inplace
```
---

## 🎵 System Audio Setup

KarmaViz can capture audio from your microphone by default, but to visualize system audio (music, videos, etc.), you need to set up audio loopback.

### Quick Setup

Run the audio setup helper:
```bash
python setup_system_audio.py
```
### Manual Setup (PulseAudio)

**Option 1: Enable Monitor Sources**
1. Install `pavucontrol`: `sudo apt install pavucontrol`
2. Open PulseAudio Volume Control
3. Go to "Recording" tab
4. Show "Monitor" sources from the dropdown
5. Set KarmaViz to record from your output device's monitor

**Option 2: Create Loopback**
```bash
# Create a loopback device
pactl load-module module-loopback latency_msec=1

# List available sources
pactl list sources short
```
**Option 3: Using ALSA Loopback**
```bash
# Load ALSA loopback module
sudo modprobe snd-aloop

# Configure ALSA to use loopback
echo "pcm.!default { type plug slave.pcm \"hw:Loopback,0,0\" }" >> ~/.asoundrc
```
### Troubleshooting Audio Issues

**No audio devices detected:**
- Check if PulseAudio is running: `pulseaudio --check -v`
- Restart PulseAudio: `pulseaudio -k && pulseaudio --start`
- Install audio development packages: `sudo apt install pulseaudio-dev portaudio19-dev`

**Audio capture not working:**
- Verify device permissions: Add user to `audio` group
- Check device availability: `python -c "import sounddevice; print(sounddevice.query_devices())"`
- Try different audio devices from the list shown at startup

**Taskbar icon not showing:**
- Ensure proper desktop environment integration
- Try running with: `SDL_VIDEO_X11_WMCLASS=KarmaViz python main.py`
- Check if window manager supports system tray icons

---

## 🎵 System Audio Setup

KarmaViz can capture audio from your microphone by default, but to visualize system audio (music, videos, etc.), you need to set up audio loopback.

### Quick Setup

Run the audio setup helper:
```bash
python setup_system_audio.py
```
### Manual Setup (PulseAudio)

**Option 1: Enable Monitor Sources**
1. Install `pavucontrol`: `sudo apt install pavucontrol`
2. Open PulseAudio Volume Control
3. Go to "Recording" tab
4. Show "Monitor" sources from the dropdown
5. Set KarmaViz to record from your output device's monitor

**Option 2: Create Loopback**
```bash
# Create a loopback device
pactl load-module module-loopback latency_msec=1

# List available sources
pactl list sources short
```
**Option 3: Using ALSA Loopback**
```bash
# Load ALSA loopback module
sudo modprobe snd-aloop

# Configure ALSA to use loopback
echo "pcm.!default { type plug slave.pcm \"hw:Loopback,0,0\" }" >> ~/.asoundrc
```
### Troubleshooting Audio Issues

**No audio devices detected:**
- Check if PulseAudio is running: `pulseaudio --check -v`
- Restart PulseAudio: `pulseaudio -k && pulseaudio --start`
- Install audio development packages: `sudo apt install pulseaudio-dev portaudio19-dev`

**Audio capture not working:**
- Verify device permissions: Add user to `audio` group
- Check device availability: `python -c "import sounddevice; print(sounddevice.query_devices())"`
- Try different audio devices from the list shown at startup

**Taskbar icon not showing:**
- Ensure proper desktop environment integration
- Try running with: `SDL_VIDEO_X11_WMCLASS=KarmaViz python main.py`
- Check if window manager supports system tray icons

---

## ⌨️ Keyboard Shortcuts

### 🎮 **Main Controls**
| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `TAB` | Toggle configuration menu |
| `F11` | Toggle fullscreen |
| `P` | Toggle pulse effect |
| `Numpad 5` | Toggle bounce effect |
| `I` | Toggle mouse interaction |
| `S` | Toggle spectrogram overlay |
| `W` | Cycle GPU waveforms |
| `R` | Cycle rotation modes |
| `M` | Cycle symmetry modes |
| `K` | Toggle kaleidoscope effect |
| `L` | Toggle warp-first rendering |

### 🎨 **Visual Effects**
| Key | Action |
|-----|--------|
| `T` / `Shift+T` | Increase/Decrease trail intensity |
| `G` / `Shift+G` | Increase/Decrease glow intensity |
| `F` / `Shift+F` | Increase/Decrease smoke intensity |
| `[` / `]` | Decrease/Increase pulse intensity |
| `↑` / `↓` | Increase/Decrease waveform scale |
| `Shift+↑` / `Shift+↓` | Increase/Decrease glow radius |

### ⚡ **Speed & Animation**
| Key | Action |
|-----|--------|
| `Numpad +` / `Numpad -` | Increase/Decrease animation speed |
| `Numpad *` / `Numpad /` | Increase/Decrease audio speed boost |
| `Numpad 3` / `Numpad 1` | Increase/Decrease palette speed |
| `Numpad 6` / `Numpad 4` | Increase/Decrease color cycle speed |

### 🎵 **Audio & Beat Controls**
| Key | Action |
|-----|--------|
| `.` / `,` | Increase/Decrease beats per change |
| `Numpad 9` / `Numpad 7` | Increase/Decrease beat sensitivity |
| `Numpad 8` / `Numpad 2` | Increase/Decrease bounce intensity |
| `/` | Toggle automatic transitions |

### 🗺️ **Warp Map Controls**
| Key | Action |
|-----|--------|
| `Space` | Manual warp map change |
| `Backspace` | Clear current warp map |

### 🎯 **Preset System**
| Key | Action |
|-----|--------|
| `0-9` | Load quick preset slot (0-9) |
| `Ctrl+0-9` | Save current settings to quick preset slot |

### 📊 **Performance & Debug**
| Key | Action |
|-----|--------|
| `F1` | Print performance statistics |
| `F2` | Clear performance statistics |
| `F3` | Toggle performance monitoring |
| `F4` | Show shader compilation status |

---

## 🎨 Waveform Categories

KarmaViz includes **100+ professionally crafted waveforms** organized into categories:

- **🌊 Basic** - Classic waveform patterns
- **🔬 Advanced** - Complex mathematical visualizations  
- **🌌 Cosmic** - Space-inspired ethereal effects
- **💻 Digital** - Cyberpunk and tech aesthetics
- **🧪 Experimental** - Cutting-edge visual experiments
- **🌀 Fractal** - Self-similar recursive patterns
- **🚀 Futuristic** - Sci-fi inspired designs
- **💡 Lighting** - Dynamic illumination effects
- **📐 Mathematical** - Geometric and algebraic forms
- **🏃 Motion** - Kinetic and flow-based patterns
- **🌿 Natural** - Organic and nature-inspired forms
- **🌱 Organic** - Fluid, life-like movements
- **📼 Retro** - Vintage and nostalgic styles

---

## 🗺️ Warp Map Effects

Transform your visualizations with **50+ warp maps**:

- **🎯 Basic** - Fundamental transformations
- **🌌 Cosmic** - Galactic distortions and stellar effects
- **💻 Digital** - Matrix-style and digital glitch effects
- **🌊 Distortion** - Wave and ripple transformations
- **🧪 Experimental** - Avant-garde visual experiments
- **🌀 Fractal** - Recursive geometric patterns
- **🚀 Futuristic** - Advanced sci-fi transformations
- **📐 Geometric** - Mathematical shape manipulations
- **🔢 Mathematical** - Algorithm-based distortions
- **🏃 Motion** - Dynamic movement effects
- **🌿 Organic** - Natural flow transformations
- **📼 Retro** - Classic visual effects
- **🎭 3D Transformations** - Dimensional manipulations

---

## 🎨 Color Palettes

Choose from **30+ stunning color palettes** or create your own using our intuitive editor:

- **🌈 Rainbow** - Full spectrum gradients
- **🌊 Ocean Themes** - Deep blues and aqua tones
- **🔥 Fire & Energy** - Warm reds, oranges, and yellows
- **🌸 Pastel Dreams** - Soft, dreamy color combinations
- **🌙 Night Sky** - Dark blues with stellar accents
- **🍃 Nature** - Earth tones and forest greens
- **💎 Precious Metals** - Gold, silver, and copper
- **🌺 Floral** - Vibrant flower-inspired palettes
- **🎭 Neon** - Electric and cyberpunk colors
- **🏔️ Arctic** - Cool blues and icy whites

---

## 🛠️ Advanced Features

### 📝 **Live Shader Editor**
- Real-time GLSL compilation
- Syntax highlighting with error detection
- Live preview with automatic updates
- Template system for quick starts
- Error reporting with line numbers

### 🎵 **Waveform Editor**
- Visual waveform design interface
- Mathematical function support
- Live audio-reactive preview
- Category organization system
- Export/import functionality

### 🗺️ **Warp Map Editor**
- 3D transformation designer
- Real-time distortion preview
- Mathematical expression support
- Complex effect layering
- Performance optimization tools

### 💾 **Preset Management**
- Complete state saving/loading
- Quick-access slot system (0-9)
- Automatic shader compilation
- Configuration export/import
- Backup and restore functionality

---

## 🔧 Configuration

### 🖥️ **Display Settings**
- **Resolution**: ALl monitor supported resolutions.
- **FPS**: 20-120
- **Anti-aliasing**: FXAA post-processing
- **Multi-monitor**: Primary/secondary display selection

### 🎵 **Audio Settings**
- **Input Device**: Uses system default output, or select alternate input in audio settings
- **Sample Rate**: 44.1kHz, 48kHz, 96kHz
- **Buffer Size**: Configurable for latency optimization
- **Beat Detection**: Sensitivity and threshold adjustment

### 🎨 **Visual Settings**
- **Effect Intensities**: Individual control for all effects
- **Color Management**: Palette speed and transition settings
- **Animation Speed**: Global and per-preset timing
- **Quality Settings**: Performance vs. visual quality balance

---

## 🚀 Performance

KarmaViz is optimized for high performance:

- **GPU Acceleration**: All rendering on graphics card
- **Threaded Compilation**: Background shader processing
- **Cython Extensions**: Optimized color operations
- **Memory Management**: Efficient buffer handling

**Recommended Specs:**
- **CPU**: Intel i5-8400 / AMD Ryzen 5 2600 or better
- **GPU**: GTX 1060 / RX 580 or better (OpenGL 3.3+) with 2GB+ Graphics Memory
- **Graphics Drivers**: Latest stable releases
- **RAM**: 8GB+ (16GB recommended)
- **Storage**: SSD recommended for shader loading
- **OS**: Linux (Ubuntu/Debian recommended)

---

## 🐛 Troubleshooting

### Common Issues

**Audio not working:**
```bash
# Linux: Check audio devices
python -c "import sounddevice; print(sounddevice.query_devices())"

# Install additional audio libraries if needed
sudo apt install pulseaudio-dev portaudio19-dev
```
**OpenGL errors:**
```bash
# Update graphics drivers
# Linux: Install mesa-utils
sudo apt install mesa-utils
glxinfo | grep "OpenGL version"
```
**Performance issues:**
- Lower FPS limit in settings
- Disable anti-aliasing
- Reduce effect intensities
- Close other GPU-intensive applications

**Shader compilation errors:**
- Check GPU OpenGL version (3.3+ required)
- Update graphics drivers
- Try different waveforms/warp maps

---

## 🤝 Contributing

KarmaViz welcomes contributions! Here's how you can help:

### 🎨 **Create Content**
- Design new waveforms (GLSL)
- Create warp map effects
- Develop color palettes
- Share preset configurations

### 🐛 **Report Issues**
- Bug reports with system information
- Performance optimization suggestions
- Feature requests and ideas
- Documentation improvements

### 💻 **Code Contributions**
- Performance optimizations
- New visual effects
- Audio processing improvements
- Cross-platform compatibility

**Development Setup:**
```bash
git clone https://github.com/yourusername/karmaviz.git
cd karmaviz
pip install -e .
python -m pytest tests/  # Run tests
```

---

## 📄 License

KarmaViz is licensed for **Personal Use Only**. 

### ✅ **Permitted Uses**
- Personal entertainment and visualization
- Educational purposes and learning
- Personal creative projects
- Private demonstrations

### ❌ **Prohibited Uses**
- Commercial performances or events
- Public performances or exhibitions
- Distribution of modified versions
- Any revenue-generating activities

For commercial licensing, please contact: **karma@karmaviz.biz**

See [LICENSE.md](LICENSE.md) for complete terms.

---

## ☕ Support the Project

**KarmaViz represents hundreds of hours of passionate development work!** 

This project features:
- 🎨 **100+ hand-crafted waveforms** with mathematical precision
- 🗺️ **50+ custom warp maps** for stunning 3D effects  
- 🎵 **Advanced audio processing** with real-time beat detection
- 🛠️ **Professional editing tools** with live preview capabilities
- ⚡ **GPU-optimized rendering** for smooth 60+ FPS performance
- 🎮 **Intuitive controls** with comprehensive keyboard shortcuts
- 🎨 **30+ color palettes** designed by a visual artist
- 📝 **Live shader compilation** with error reporting
- 💾 **Complete preset system** for saving your creations

### 🙏 **Show Your Appreciation**

If KarmaViz has enhanced your music experience or inspired your creativity, consider supporting its continued development:

<div align="center">

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support%20KarmaViz-orange?style=for-the-badge&logo=buy-me-a-coffee&logoColor=white)](https://buymeacoffee.com/karmaviz)

**[☕ Buy me a coffee](https://buymeacoffee.com/karmaviz)**

</div>

Your support helps:
- 🚀 **Accelerate development** of new features
- 🎨 **Create more visual content** (waveforms, effects, palettes)
- 🐛 **Maintain and improve** existing functionality  
- 📚 **Expand documentation** and tutorials
- 🌍 **Support the open-source community**

### 💝 **Other Ways to Support**
- ⭐ **Star this repository** to show your appreciation
- 🐛 **Report bugs** and suggest improvements
- 🎨 **Share your creations** and presets with the community
- 📢 **Spread the word** about KarmaViz to fellow music lovers
- 💻 **Contribute code** or documentation improvements

---

## 🌟 Acknowledgments

**Special thanks to:**
- The **ModernGL** community for excellent OpenGL bindings
- **PyQt5** developers for the robust GUI framework
- **NumPy/SciPy** teams for powerful numerical computing
- **Pygame** community for multimedia support
- All **beta testers** and **contributors** who helped shape KarmaViz

---

## 📞 Contact

- **🐛 Issues & Bugs**: [GitHub Issues](https://github.com/KarmaTripping/KarmaViz/issues)
all -e .
python -m pytest tests/  # Run tests
```

---

## 📄 License

KarmaViz is licensed for **Personal Use Only**. 

### ✅ **Permitted Uses**
- Personal entertainment and visualization
- Educational purposes and learning
- Personal creative projects
- Private demonstrations

### ❌ **Prohibited Uses**
- Commercial performances or events
- Public performances or exhibitions
- Distribution of modified versions
- Any revenue-generating activities

For commercial licensing, please contact: **karma@karmaviz.biz**

See [LICENSE.md](LICENSE.md) for complete terms.

---

## ☕ Support the Project

**KarmaViz represents hundreds of hours of passionate development work!** 

This project features:
- 🎨 **100+ hand-crafted waveforms** with mathematical precision
- 🗺️ **50+ custom warp maps** for stunning 3D effects  
- 🎵 **Advanced audio processing** with real-time beat detection
- 🛠️ **Professional editing tools** with live preview capabilities
- ⚡ **GPU-optimized rendering** for smooth 60+ FPS performance
- 🎮 **Intuitive controls** with comprehensive keyboard shortcuts
- 🎨 **30+ color palettes** designed by a visual artist
- 📝 **Live shader compilation** with error reporting
- 💾 **Complete preset system** for saving your creations

### 🙏 **Show Your Appreciation**

If KarmaViz has enhanced your music experience or inspired your creativity, consider supporting its continued development:

<div align="center">

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support%20KarmaViz-orange?style=for-the-badge&logo=buy-me-a-coffee&logoColor=white)](https://buymeacoffee.com/karmaviz)

**[☕ Buy me a coffee](https://buymeacoffee.com/karmaviz)**

</div>

Your support helps:
- 🚀 **Accelerate development** of new features
- 🎨 **Create more visual content** (waveforms, effects, palettes)
- 🐛 **Maintain and improve** existing functionality  
- 📚 **Expand documentation** and tutorials
- 🌍 **Support the open-source community**

### 💝 **Other Ways to Support**
- ⭐ **Star this repository** to show your appreciation
- 🐛 **Report bugs** and suggest improvements
- 🎨 **Share your creations** and presets with the community
- 📢 **Spread the word** about KarmaViz to fellow music lovers
- 💻 **Contribute code** or documentation improvements

---

## 🌟 Acknowledgments

**Special thanks to:**
- The **ModernGL** community for excellent OpenGL bindings
- **PyQt5** developers for the robust GUI framework
- **NumPy/SciPy** teams for powerful numerical computing
- **Pygame** community for multimedia support
- All **beta testers** and **contributors** who helped shape KarmaViz

---

## 📞 Contact

- **🐛 Issues & Bugs**: [GitHub Issues](https://github.com/KarmaTripping/KarmaViz/issues)
 welcomes contributions! Here's how you can help:

### 🎨 **Create Content**
- Design new waveforms (GLSL)
- Create warp map effects
- Develop color palettes
- Share preset configurations

### 🐛 **Report Issues**
- Bug reports with system information
- Performance optimization suggestions
- Feature requests and ideas
- Documentation improvements

### 💻 **Code Contributions**
- Performance optimizations
- New visual effects
- Audio processing improvements
- Cross-platform compatibility

**Development Setup:**
```bash
git clone https://github.com/yourusername/karmaviz.git
cd karmaviz
pip install -e .
python -m pytest tests/  # Run tests
```

---

## 📄 License

KarmaViz is licensed for **Personal Use Only**. 

### ✅ **Permitted Uses**
- Personal entertainment and visualization
- Educational purposes and learning
- Personal creative projects
- Private demonstrations

### ❌ **Prohibited Uses**
- Commercial performances or events
- Public performances or exhibitions
- Distribution of modified versions
- Any revenue-generating activities

For commercial licensing, please contact: **karma@karmaviz.biz**

See [LICENSE.md](LICENSE.md) for complete terms.

---

## ☕ Support the Project

**KarmaViz represents hundreds of hours of passionate development work!** 

This project features:
- 🎨 **100+ hand-crafted waveforms** with mathematical precision
- 🗺️ **50+ custom warp maps** for stunning 3D effects  
- 🎵 **Advanced audio processing** with real-time beat detection
- 🛠️ **Professional editing tools** with live preview capabilities
- ⚡ **GPU-optimized rendering** for smooth 60+ FPS performance
- 🎮 **Intuitive controls** with comprehensive keyboard shortcuts
- 🎨 **30+ color palettes** designed by a visual artist
- 📝 **Live shader compilation** with error reporting
- 💾 **Complete preset system** for saving your creations

### 🙏 **Show Your Appreciation**

If KarmaViz has enhanced your music experience or inspired your creativity, consider supporting its continued development:

<div align="center">

[![Buy Me A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support%20KarmaViz-orange?style=for-the-badge&logo=buy-me-a-coffee&logoColor=white)](https://buymeacoffee.com/karmaviz)

**[☕ Buy me a coffee](https://buymeacoffee.com/karmaviz)**

</div>

Your support helps:
- 🚀 **Accelerate development** of new features
- 🎨 **Create more visual content** (waveforms, effects, palettes)
- 🐛 **Maintain and improve** existing functionality  
- 📚 **Expand documentation** and tutorials
- 🌍 **Support the open-source community**

### 💝 **Other Ways to Support**
- ⭐ **Star this repository** to show your appreciation
- 🐛 **Report bugs** and suggest improvements
- 🎨 **Share your creations** and presets with the community
- 📢 **Spread the word** about KarmaViz to fellow music lovers
- 💻 **Contribute code** or documentation improvements

---

## 🌟 Acknowledgments

**Special thanks to:**
- The **ModernGL** community for excellent OpenGL bindings
- **PyQt5** developers for the robust GUI framework
- **NumPy/SciPy** teams for powerful numerical computing
- **Pygame** community for multimedia support
- All **beta testers** and **contributors** who helped shape KarmaViz

---

## 📞 Contact

- **🐛 Issues & Bugs**: [GitHub Issues](https://github.com/KarmaTripping/KarmaViz/issues)
- **💼 Commercial Licensing**: karma@karmaviz.biz
- **☕ Support Development**: [Buy Me A Coffee](https://buymeacoffee.com/karmaviz)
- **💬 Feedback** and Suggestions: karma@karmaviz.biz
---

<div align="center">

**Made with ❤️ and countless hours of dedication**

*Transform your music into mesmerizing visual art with KarmaViz*

[![Buy Me A Coffee](https://img.shields.io/badge/☕-Support%20This%20Project-orange?style=flat-square)](https://buymeacoffee.com/karmaviz)

</div>
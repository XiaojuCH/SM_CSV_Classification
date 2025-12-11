# CMake 构建指南 - 光学特征分类器

## 📚 CMake 学习 + 项目实战

这个项目是学习CMake的绝佳实例！我会边讲解CMake概念，边展示如何构建项目。

---

## 🎯 快速开始（3种方法）

### 方法1：使用Python脚本（推荐，最简单）

```bash
# 默认构建（Release模式）
python build.py

# 清理后重新构建
python build.py --clean

# Debug模式构建
python build.py --config Debug

# 查看所有选项
python build.py --help
```

### 方法2：使用CMake命令行（学习CMake）

```bash
# 1. 配置项目（生成构建文件）
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_DIR=C:/onnxruntime

# 2. 编译项目
cmake --build build --config Release

# 3. 安装（可选）
cmake --install build --prefix C:/MyInstallPath
```

### 方法3：使用CMake GUI（可视化）

1. 打开 CMake GUI
2. 设置源代码路径：`CPP文件夹路径`
3. 设置构建路径：`CPP/build`
4. 点击 "Configure"
5. 选择生成器（如 Visual Studio 17 2022）
6. 点击 "Generate"
7. 点击 "Open Project" 在Visual Studio中打开

---

## 📖 CMake 基础概念讲解

### 1. CMakeLists.txt 是什么？

CMakeLists.txt 是CMake的配置文件，类似于：
- Makefile（Make工具）
- build.gradle（Gradle）
- package.json（npm）

**作用**：告诉CMake如何构建你的项目。

### 2. CMake 构建流程

```
源代码 + CMakeLists.txt
         ↓
    [cmake 配置]  ← 生成构建文件（Makefile/VS项目等）
         ↓
    [cmake 编译]  ← 调用编译器编译
         ↓
      可执行文件/库
```

### 3. 关键CMake命令

#### `cmake_minimum_required(VERSION 3.15)`
指定最低CMake版本，确保兼容性。

#### `project(OpticalClassifier VERSION 1.0.0)`
定义项目名称和版本。

#### `set(CMAKE_CXX_STANDARD 17)`
设置C++标准（C++17）。

#### `add_library(ClassifierDLL SHARED ClassifierDLL.cpp)`
创建一个共享库（DLL）目标。
- `SHARED` = 动态库（.dll/.so）
- `STATIC` = 静态库（.lib/.a）

#### `target_link_libraries(ClassifierDLL PRIVATE ${ONNXRUNTIME_LIB})`
链接外部库到目标。

#### `include_directories(${ONNXRUNTIME_DIR}/include)`
添加头文件搜索路径。

---

## 🔍 我们的 CMakeLists.txt 详解

让我逐段解释我们的CMakeLists.txt：

### 第1部分：项目基本配置

```cmake
cmake_minimum_required(VERSION 3.15)
project(OpticalClassifier VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

**解释**：
- 要求CMake 3.15或更高版本
- 项目名称：OpticalClassifier，版本1.0.0
- 使用C++17标准

### 第2部分：查找ONNX Runtime

```cmake
if(NOT DEFINED ONNXRUNTIME_DIR)
    if(DEFINED ENV{ONNXRUNTIME_DIR})
        set(ONNXRUNTIME_DIR $ENV{ONNXRUNTIME_DIR})
    else()
        set(ONNXRUNTIME_DIR "C:/onnxruntime")
    endif()
endif()
```

**解释**：
- 支持3种方式指定ONNX Runtime路径：
  1. 命令行参数：`-DONNXRUNTIME_DIR=路径`
  2. 环境变量：`set ONNXRUNTIME_DIR=路径`
  3. 默认路径：`C:/onnxruntime`

**CMake变量优先级**：命令行 > 环境变量 > 默认值

### 第3部分：检查依赖

```cmake
if(NOT EXISTS "${ONNXRUNTIME_DIR}/include")
    message(FATAL_ERROR "ONNX Runtime not found...")
endif()
```

**解释**：
- `message(FATAL_ERROR ...)` 会停止CMake并显示错误
- 确保ONNX Runtime存在才继续

### 第4部分：创建DLL目标

```cmake
add_library(ClassifierDLL SHARED ClassifierDLL.cpp)
target_link_libraries(ClassifierDLL PRIVATE ${ONNXRUNTIME_LIB})
```

**解释**：
- `add_library` 创建库目标
- `SHARED` 表示动态库（DLL）
- `target_link_libraries` 链接ONNX Runtime库
- `PRIVATE` 表示链接关系不传递给依赖者

### 第5部分：复制依赖文件

```cmake
add_custom_command(TARGET ClassifierDLL POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${ONNXRUNTIME_DLL}"
    $<TARGET_FILE_DIR:ClassifierDLL>
)
```

**解释**：
- `POST_BUILD` 在构建完成后执行
- `copy_if_different` 只在文件不同时复制（提高效率）
- `$<TARGET_FILE_DIR:...>` 是生成器表达式，获取目标输出目录

### 第6部分：设置输出目录

```cmake
set_target_properties(ClassifierDLL PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
```

**解释**：
- 统一输出目录，方便管理
- `CMAKE_BINARY_DIR` 是构建目录（build/）

---

## 🛠️ CMake 常用命令

### 配置阶段

```bash
# 基本配置
cmake -S <源码目录> -B <构建目录>

# 指定生成器
cmake -S . -B build -G "Visual Studio 17 2022"

# 指定构建类型
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 设置变量
cmake -S . -B build -DONNXRUNTIME_DIR=C:/onnxruntime

# 查看所有变量
cmake -S . -B build -L

# 查看详细变量
cmake -S . -B build -LA
```

### 编译阶段

```bash
# 编译所有目标
cmake --build build

# 指定配置（多配置生成器如VS）
cmake --build build --config Release

# 并行编译
cmake --build build --parallel 8

# 编译特定目标
cmake --build build --target ClassifierDLL

# 清理
cmake --build build --target clean
```

### 安装阶段

```bash
# 安装到默认位置
cmake --install build

# 安装到指定位置
cmake --install build --prefix C:/MyApp

# 指定配置
cmake --install build --config Release
```

---

## 🎓 CMake 进阶技巧

### 1. 生成器表达式

生成器表达式在生成构建文件时求值，语法：`$<...>`

```cmake
# 获取目标文件目录
$<TARGET_FILE_DIR:ClassifierDLL>

# 根据配置选择不同值
$<$<CONFIG:Debug>:debug_flag>

# 根据平台选择
$<$<PLATFORM_ID:Windows>:windows_specific>
```

### 2. 条件编译

```cmake
if(WIN32)
    # Windows特定代码
elseif(UNIX)
    # Linux/Mac特定代码
endif()
```

### 3. 查找包

```cmake
# 查找包
find_package(OpenCV REQUIRED)

# 使用包
target_link_libraries(MyTarget PRIVATE OpenCV::OpenCV)
```

### 4. 选项

```cmake
# 定义选项
option(BUILD_TESTS "Build tests" ON)

# 使用选项
if(BUILD_TESTS)
    add_subdirectory(tests)
endif()
```

---

## 📂 构建目录结构

运行CMake后的目录结构：

```
CPP/
├── CMakeLists.txt           # CMake配置文件
├── ClassifierDLL.cpp        # 源代码
├── build/                   # 构建目录（生成）
│   ├── CMakeCache.txt       # CMake缓存
│   ├── CMakeFiles/          # CMake内部文件
│   ├── bin/                 # 可执行文件输出
│   │   └── Release/
│   │       ├── ClassifierDLL.dll
│   │       ├── onnxruntime.dll
│   │       └── *.onnx, *.json
│   └── lib/                 # 库文件输出
│       └── Release/
│           └── ClassifierDLL.lib
```

---

## 🔧 实战练习

### 练习1：修改输出目录

在CMakeLists.txt中修改输出目录：

```cmake
set_target_properties(ClassifierDLL PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/output
)
```

然后重新构建：
```bash
python build.py --clean
```

### 练习2：添加编译选项

添加编译器警告：

```cmake
if(MSVC)
    target_compile_options(ClassifierDLL PRIVATE /W4)
else()
    target_compile_options(ClassifierDLL PRIVATE -Wall -Wextra)
endif()
```

### 练习3：添加预处理宏

```cmake
target_compile_definitions(ClassifierDLL PRIVATE
    VERSION_MAJOR=1
    VERSION_MINOR=0
)
```

在C++代码中可以使用：
```cpp
#ifdef VERSION_MAJOR
    std::cout << "Version: " << VERSION_MAJOR << "." << VERSION_MINOR << std::endl;
#endif
```

---

## 🐛 常见问题

### Q1: CMake找不到

**错误**：`'cmake' is not recognized...`

**解决**：
1. 下载CMake：https://cmake.org/download/
2. 安装时选择"Add CMake to system PATH"
3. 或手动添加到PATH：`C:/Program Files/CMake/bin`

### Q2: 找不到ONNX Runtime

**错误**：`ONNX Runtime not found at: C:/onnxruntime`

**解决**：
```bash
# 方法1：运行下载脚本
download_onnxruntime.bat

# 方法2：指定路径
python build.py --onnxruntime-dir D:/MyONNXRuntime
```

### Q3: 生成器不匹配

**错误**：`The C compiler identification is unknown`

**解决**：
```bash
# 明确指定生成器
python build.py --generator "Visual Studio 17 2022"

# 或使用Ninja（更快）
python build.py --generator "Ninja"
```

### Q4: 多配置生成器问题

Visual Studio是多配置生成器，需要指定配置：

```bash
# 错误方式
cmake --build build

# 正确方式
cmake --build build --config Release
```

---

## 📚 CMake 学习资源

### 官方文档
- [CMake官方教程](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
- [CMake命令参考](https://cmake.org/cmake/help/latest/manual/cmake-commands.7.html)

### 推荐书籍
- "Professional CMake: A Practical Guide"
- "Mastering CMake"

### 在线资源
- [CMake Examples](https://github.com/ttroy50/cmake-examples)
- [Modern CMake](https://cliutils.gitlab.io/modern-cmake/)

---

## 🎯 总结

### 你学到了什么

1. ✅ **CMake基础**：项目配置、目标创建、依赖管理
2. ✅ **CMake命令**：配置、编译、安装
3. ✅ **实战技巧**：查找依赖、复制文件、设置属性
4. ✅ **Python集成**：用Python脚本简化CMake使用

### 下一步

1. **修改CMakeLists.txt**：尝试添加新的编译选项
2. **创建子项目**：学习`add_subdirectory()`
3. **使用find_package()**：学习查找和使用第三方库
4. **编写测试**：学习`enable_testing()`和`add_test()`

---

## 🚀 快速命令参考

```bash
# 配置
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# 编译
cmake --build build --config Release --parallel

# 安装
cmake --install build --prefix C:/MyApp

# 清理
rm -rf build

# 使用Python脚本（推荐）
python build.py                    # 默认构建
python build.py --clean            # 清理重建
python build.py --config Debug     # Debug构建
python build.py --install          # 构建并安装
```

---

**祝你CMake学习愉快！有问题随时问我。** 🎉

## 2025-10-26
- 梳理 `src/core/auto_aim` 下各子模块（aimer、detected、planner、pointer、solver、target、tracker）依赖，讨论合并为单一动态库的方案。
- 新建统一的 `auto_aim` 包：编写顶层 `CMakeLists.txt` 与 `package.xml`，将子模块编译为静态库后以 `--whole-archive` 聚合为一个共享库；删除原子包的 CMake 与 package 文件。
- 修复 tinympc 头文件未安装导致的构建错误，新增最小的 `auto_aim_module.cpp` 作为顶层共享库的占位源文件，确保链接成功。
- 复查构建：`colcon build --packages-select auto_aim` 通过，仅剩若干警告；总结问题原因并记录后续注意事项。

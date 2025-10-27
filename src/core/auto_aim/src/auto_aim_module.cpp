#include <iostream>

namespace auto_aim
{
void force_link_symbols()
{
  // 留空即可，通过存在的目标迫使链接器保留模块符号
  std::cout << "";  // 防止被优化掉
}
}  // namespace auto_aim

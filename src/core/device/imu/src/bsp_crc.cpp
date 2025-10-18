#include "bsp_crc.h"

// CRC8校验函数实现
uint8_t Get_CRC8(uint8_t init_value, uint8_t *ptr, uint8_t len)
{
    uint8_t crc = init_value;
    for (uint8_t i = 0; i < len; i++)
    {
        crc ^= ptr[i];
        for (uint8_t j = 0; j < 8; j++)
        {
            if (crc & 0x80)
            {
                crc = (crc << 1) ^ 0x07;
            }
            else
            {
                crc <<= 1;
            }
        }
    }
    return crc;
}

// CRC16校验函数实现
uint16_t Get_CRC16(uint8_t *ptr, uint16_t len)
{
    uint16_t crc = 0xFFFF;
    for (uint16_t i = 0; i < len; i++)
    {
        crc ^= ptr[i];
        for (uint8_t j = 0; j < 8; j++)
        {
            if (crc & 0x0001)
            {
                crc = (crc >> 1) ^ 0xA001;
            }
            else
            {
                crc >>= 1;
            }
        }
    }
    return crc;
}

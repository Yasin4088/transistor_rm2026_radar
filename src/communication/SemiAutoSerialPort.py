import sys
import serial
import serial.tools.list_ports

# 无串口时的兜底串口号：Linux 用 /dev/ttyUSB0，Windows 用 COM1
DEFAULT_PORT_FALLBACK = "/dev/ttyUSB0" if sys.platform.startswith("linux") else "COM1"

def semi_auto_serial_port(default_port = None):
    port_list = list(serial.tools.list_ports.comports())
    port_name_list = [port_info[0] for port_info in port_list]
    if len(port_list) == 0:
        if default_port:
            print(f"警告：串口列表为空，将返回默认串口[{default_port}]")
            return default_port
        else:
            print(f"警告：串口列表为空且未设置默认串口，将返回[{DEFAULT_PORT_FALLBACK}]")
            return DEFAULT_PORT_FALLBACK
    elif len(port_list) == 1:
        port_name = port_name_list[0]
        print(f"发现唯一串口[{port_name}]")
        if default_port and port_name != default_port:
            print(f"警告：默认串口[{default_port}]与发现串口[{port_name}]不一致，将返回发现串口")
        return port_name
    else:
        if default_port and default_port in port_name_list:
            print(f"发现默认串口[{default_port}]，将返回默认串口")
            return default_port
        if not default_port:
            print("警告：发现多个串口且未设置默认串口，请手动输入串口号")
        else:
            print("警告：发现多个串口且未发现默认串口，请手动输入串口号")
        print(f"发现串口列表：{port_name_list}")
        if sys.platform.startswith("linux"):
            print("提示：裁判系统 USB 转串口一般为 /dev/ttyUSB0 或 /dev/ttyACM0；直接回车可进入虚拟串口模式")
            return input("输入选择的串口号（如/dev/ttyUSB0，直接回车跳过）：")
        return input("输入选择的串口号（如COM1）：")

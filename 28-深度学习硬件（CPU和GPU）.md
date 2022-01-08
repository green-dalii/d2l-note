# 28 - 深度学习硬件CPU、GPU、TPU和其他

### 🎦 本节课程视频地址 👉[![Bilibil](	https://i2.hdslb.com/bfs/archive/8d8c0bed24f9fd3c760c3ca34ce992061824f371.jpg@640w_400h_100Q_1c.webp)](https://www.bilibili.com/video/BV1TU4y1j7Wd)

[![Bilibil](https://i1.hdslb.com/bfs/archive/6c4b1968346e5e66314f48311991b34b26244670.jpg@640w_400h_100Q_1c.webp)](https://www.bilibili.com/video/BV1VV41147PC)

以一台电脑配置为例：
- CPU: Intel i7, 0.15TFLOPS
- DDR4: 32GB
- GPU：Nvidia Titan X, 12TFLOPS, 16GB显存

![](\Images/CPU-vs-GPU.jpg)

**提升CPU利用率**

- 在计算$a+b$之前，需要准备数据
  - 主内存->L3(LLC一般指逻辑链路控制)->L2->L1->寄存器——每级cache逐渐变小。
    - L1访问延时：0.5ns
    - L2访问延时：7ns(14xL1)
    - 主内存访问延时：100ns(200xL1)
    - 不断访问造成时间浪费
- 提升空间和时间的内存本地性
  - 时间：重用数据使得保持它们在缓存里
  - 空间：按序读写数据使得可以预读取

**样例分析**

如果一个矩阵是按列存储，访问一行会比访问一列要快
- CPU一次读取64字节（缓存线）
- CPU会“聪明地”提前读取下一个（缓存线）

**提升CPU利用率II**

- 高端CPU有几十个核
- 并行来利用所有核
  - 超线程不一定能提升性能，因为他们共享寄存器

**样例分析**

```
for i in range(len(a)):
    c[i] = a[i] + b[i]
```
<center> <font color=red> VS</font></center>
```
import numpy
c = a + b
```
- 左边调用n次函数，每次调用有开销
- 右边很容易被并行

**CPU VS GPU**

![](\Images/1_L9SPSTIq_ptT6a5ejgzmAQ.png)

- CPU一个核一般只有一个计算单元，一般6-64核，30-100GB/s内存带宽，控制流强
- GPU一个核可以有几十个计算单元，且核多，动辄2k-4k核，水涨船高，400GB/s-1TB/s，控制流强

![](\Images/01-cpugpuarch.png)

**提升GPU利用率**

- 并行
  - 使用数千个线程
  - 所以小的神经网络反而不需要此“牛刀”
- 内存本地性
  - 缓存更小，架构更加简单
  - GPU为了节省面积，所以缓存面积很小
- 少用控制语句
  - 支持有限
  - 同步开销很大

**CPU/GPU带宽**

- 不要频繁在CPU和GPU之间传数据：带宽限制，同步开销
  - 数据传递速度受限GB/s
  - 需要CG同步

**更多的CPUs和GPUs**

- CPU: AMD, ARM
- GPU: AMD. Intel, ARM, Qualcomm...

**GPU/CPU高性能计算**

- CPU: C++或者任何高性能语言
  - 编译器成熟
- GPU:
  - Nvidia上用CUDA
    - 编译器和驱动成熟
  - 其他用OpenCL
    - 质量取决于硬件厂商

**总结**

- CPU：可以处理通用计算。性能优化考虑数据读写效率和多线程
- CPU：使用更多的小核和更好的内存带宽，适合能大规模并行的计算任务

### TPU和其他

**DSP**
- 为数字信号处理算法设计：点积，卷积，FFT
- 低功耗、高性能（核不多，频率低）
  - 比移动GPU快5x，功耗更低
- VLIW: Very long instruction word
  - 一条指令计算上百次乘累加
- 编程和调试困难
- 编译器质量良莠不齐

待续吧……
### 数据并行和模型并行

- 数据并行：把小批量分成n块，每个GPU拿到完整参数计算一块数据的梯度
  - 通常性能更好

![](\Images/Data-parallelism-approach-of-DACDRP.png)
- 模型并行：将模型分成n块，每个GPU拿到他的一部分模型计算前向、后向计算
  - 针对特别大的模型，几百GB

**总结**

- 当一个模型能用单卡计算时，通常使用数据并行拓展到更多卡上
- 模型并行则用在超大模型上
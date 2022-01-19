# 番外 03 - Linux 常用命令

## 开启/禁用 Swap 交换文件

- 检查是否开启 swapfile

```bash
free -m
# 如果swap一行显示：
# swap:  0   0   0
# 代表没有开启
```

- 开启 Swap

```bash
# 在系统根目录创建swapfile文件，指定大小为4GB
dd if=/dev/zero of=/swapfile count=4096 bs=1M
# 激活swapfile
chmod 600 /swapfile
mkswap /swapfile

# 看到如下结果证明激活成功：
# Setting up swapspace version 1, size = 4096000 KiB
# no label, UUID=ff3fc469-9c4b-4913-b653-ec53d6460d0e

# 开启Swap
swapon /swapfile

# 开机自动挂载swap
echo "/swap swap swap defaults    0  0" >> /etc/fstab

# 调整Linux内核swappiness值，值越大表示越积极使用swap分区，越小表示越积极使用物理内存
# 比如30代表当内存使用至100%-30%=70%时，开始使用swap
#查看当前swappiness值
cat /proc/sys/vm/swappiness
# 修改swappiness值
sysctl vm.swappiness=10
# 开机自动修改
echo "vm.swappiness=10" >> /etc/sysctl.conf
```

- 停用 Swap

```bash
swapoff -v /swapfile

# 然后在/etc/fstab文件中删除“/swap swap swap defaults    0  0”一行

# 删除swapfile文件
rm /swapfile
```

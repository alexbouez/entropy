# Kernel compilation for Ubuntu

## Cloning

```console
sudo apt install git
git clone https://gitlab.science.ru.nl/abouez/phd-prng.git
```

## Compiling the kernel for x86

> Instructions from [linux.com](linux.com/topic/desktop/how-compile-linux-kernel-0)

Extract kernel from phd-prng/Projects/kernel-analysis/kernel/src/linux-6.0.5.tar.xz

```console
cd linux-6.0.5
sudo apt install build-essential fakeroot ncurses-dev xz-utils libssl-dev bc flex libelf-dev bison

cp /boot/config-$(uname -r) .config

make
sudo make modules_install
sudo make install

sudo update-initramfs -c -k 6.0.5
sudo update-grub
sudo reboot 0
```

Make commands a lot quicker by adding 
```console
make -j <#cpu(s)>
```

First Make step requires answering many questions > all set to default answer (pressing enter)

/!\ Updating from kernel 5.15 (ubuntu 22.04) to version 6.0.5 will give the error: 
> No rule to make target 'debian/canonical-certs.pem' 
Can be solved using :
```console
scripts/config --disable SYSTEM_TRUSTED_KEYS
scripts/config --disable SYSTEM_REVOCATION_KEYS
```

## Compiling the kernel for ARM RPi 

> Instructions from [raspberrypi.com](https://www.raspberrypi.com/documentation/computers/linux_kernel.html)

Extract kernel from phd-prng/Projects/kernel-analysis/kernel/src/linux-rpi-5.15.tar.xz
(or download from [Raspberry Pi github](https://github.com/raspberrypi/linux))


```console
cd linux-rpi-5.15.0
sudo apt install bc bison flex libssl-dev build-essential
``` 

## Compilation for 32 bits

```console
KERNEL=kernel71
make bcm2711_defconfig

CONFIG_LOCALVERSION="-v71-trace_printk"
make -j4 zImage modules dtbs

sudo make modules_install 

sudo cp arch/arm64/boot/dts/broadcom/*.dtb /boot/
sudo cp arch/arm64/boot/dts/overlays/*.dtb* /boot/overlays/
sudo cp arch/arm64/boot/dts/overlays/README /boot/overlays/
sudo cp arch/arm64/boot/zImage /boot/$KERNEL.img

sudo cp arch/arm64/boot/dts/broadcom/*.dtb /boot/firmware/
sudo cp arch/arm64/boot/dts/overlays/*.dtb* /boot/firmware/overlays/
sudo cp arch/arm64/boot/dts/overlays/README /boot/firmware/overlays/
sudo cp arch/arm64/boot/zImage /boot/firmware/$KERNEL.img
``` 

In /boot/firmware/config.txt, modify to "kernel=kernel71.img". Then, 

```console
sudo cp /boot/firmware/config.txt /boot/
sudo reboot 0
``` 

## Compilation for 64 bits

```console
KERNEL=kernel8
make bcm2711_defconfig

CONFIG_LOCALVERSION="-v8-trace_printk"
make -j4 Image.gz modules dtbs

sudo make modules_install 

sudo cp arch/arm64/boot/dts/broadcom/*.dtb /boot/
sudo cp arch/arm64/boot/dts/overlays/*.dtb* /boot/overlays/
sudo cp arch/arm64/boot/dts/overlays/README /boot/overlays/
sudo cp arch/arm64/boot/Image.gz /boot/$KERNEL.img

sudo cp arch/arm64/boot/dts/broadcom/*.dtb /boot/firmware/
sudo cp arch/arm64/boot/dts/overlays/*.dtb* /boot/firmware/overlays/
sudo cp arch/arm64/boot/dts/overlays/README /boot/firmware/overlays/
sudo cp arch/arm64/boot/Image.gz /boot/firmware/$KERNEL.img
``` 

In /boot/firmware/config.txt, modify kernel value to "kernel=kernel8.img". Then, 

```console
sudo cp /boot/firmware/config.txt /boot/
sudo reboot 0
``` 

## Setting up ring buffer extraction

Difference /sys/kernel/tracing/trace & /sys/kernel/debug/tracing/trace ?

We use a modified kernel and the trace_printk tool. 
This writes to a ring buffer readable at /sys/kernel/tracing/trac1111111111111111111111111111111111111111111111111111111111111111111111111111111e.

See current size per core (in kb): 
```console
sudo cat /sys/kernel/trace/buffer_size_kb
```

See current RAM capabilities and usage: 
```console
cat /proc/meminfo
free
```

### Bigger buffer space with grub

```console
sudo vim /etc/default/grub
```

Add the following line to the GRUB_CMDLINE_LINUX section: 
```
trace_buf_size=(size in bytes)
```
Value is in bytes, so 144384000b will give 141000kb in /sys/kernel/trace/buffer_size_kb.

### Bigger buffer space with systemctl

Create a script "/boot/set_buffer_size.sh" :
```bash
#!/bin/bash
echo (size in kb) > /sys/kernel/tracing/buffer_size_kb
```

And a file "/etc/systemd/system/set_buffer_size.service":
```
[Unit]
Description=Set buffer size

[Service]
ExecStart=/path/to/set_buffer_size.sh

[Install]
WantedBy=multi-user.target
```

Then, 
```console
chmod +x /boot/set_buffer_size.sh
sudo systemctl daemon-reload
sudo systemctl enable set_buffer_size.service
sudo systemctl start set_buffer_size.service
sudo systemctl status set_buffer_size.service
```

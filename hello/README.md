需使用VSCode,并安装Mojo扩展插件

按照[https://developer.modular.com/download](https://developer.modular.com/download)官网的教程，设置VSCode容器开发配置

一切就绪后，进入容器输入以下命令执行示例

```bash
mojo hello.mojo
```

### 注意

因为Ubuntu系统下pip安装会遇到externally-managed-environment错误，所以本机mojo不太容易安装成功，比较适合容器下开发
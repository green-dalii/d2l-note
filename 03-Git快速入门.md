# 03 - Git快速入门

---


## 1.Git的安装与VSCode集成配置
- Git下载地址：https://git-scm.com/downloads
- 安装完毕后打开命令提示符（CMD）或终端(Terminal)，输入：`git --version`，若能正常显示Git版本，则代表安装成功
- VSCode会自动与安装的Git进行关联，在已有的Git项目中，会自动追踪文件的更改，点击VSCode界面左侧第三个的Git图标将会显示以下界面：

![](https://user-images.githubusercontent.com/1217769/29250412-5b01d194-8042-11e7-90e6-a03e0d78af41.gif)

## 2.Git基本语法
1. **初始化本地仓库**：用于创建本地新仓库
   1. 首先使用CMD或Terminal进入项目文件夹 *(使用`cd "path/folder"`命令)*
   2. 输入：**`git init`**
   3. 执行完毕后，目录下将会生成一个名为`.git`的隐藏文件夹，用于Git的处理记录，至此该文件夹下的所有文件的修改将被Git追踪
   
2. **克隆远程仓库**：用于下载Github等代码托管平台已有的项目到本地
   1. 首先在代码托管平台复制想要克隆的项目git地址，Github平台git地址位置如下：
   ![Github](https://docs.github.com/assets/images/help/repository/https-url-clone.png)
   Gitee平台git地址位置如下
   ![Gitee](https://images.gitee.com/uploads/images/2018/0815/115602_7e40b5ff_551147.png "Gitee")
   2. 打开终端进入需要存储的位置，输入：**`git clone https://github.com/xxxxx/xxx.git`**
   3. 命令执行完毕后，该文件夹下会生成与远程仓库一致的所有文件
3. **其他命令**：由于后期主要涉及在VSCode上进行UI操作，其他命令请参考本文第四节：**参考资源**
## 3.使用Gitee进行多人在线协同
1. 首先需要每个成员注册各自的Gitee账户，步骤参考👉[此处](https://gitee.com/help/articles/4113)
2. 发起方需要在Gitee上创建一个新仓库，步骤参考👉[此处](https://gitee.com/help/articles/4120)
3. 发起方在仓库管理界面将需要加入协同的其他用户添加至项目的『开发者』，步骤参考👉[此处](https://gitee.com/help/articles/4175)
4. 所有用户按照本文第二节中的步骤Clone新仓库至本地
5. 使用VSCode打开本地仓库文件夹，在VSCode底部的终端输入以下命令，绑定Gitee账户并指定远程仓库地址：
   ```
   git config --global user.name "Gitee账户的用户名"
   git config --global user.email "Gitee账户的注册邮箱"
   git remote add origin https://gitee.com/xxxxx/xxx.git
   ```
1. **注意！！！** 在每次协同工作前，为确保本地仓库与远程仓库保持最新的一致性，必须先进行**拉取**(Pull)操作，点击“源代码管理”面板右上角菜单，选择“拉取”，
   ![](https://cache.yisu.com/upload/information/20210222/263/2395.jpg)
2. 如果在首次拉取时，报错`fatal: refusing to merge unrelated histories`，则在VSCode终端中输入：`git pull origin master --allow-unrelated-histories`
3. 接下来便可进行协同工作，在VSCode打开的当前仓库文件夹下，创建或修改文件并**保存**，VSCode集成的Git插件会自动跟踪文件修改，并在源代码管理界面列出所有文件更改。
4. 每次修改完在界面左上方的消息框中，输入本次的**Commit**，即修改说明，Commit指南见👉[此处](https://gitee.com/help/articles/4231)，填写完毕后点击旁边的✔，提交更改，此时VSCode将在后台执行一系列命令，自动化完成操作：
    ![](https://user-images.githubusercontent.com/1217769/29250412-5b01d194-8042-11e7-90e6-a03e0d78af41.gif)
5. 至此对本地仓库的修改已生效，要使所有成员都能看到你的更改，还需要提交至远程仓库，方法为点击“源代码管理”面板右上角菜单，选择“推送”
6. 所有成员将会看到你的更改

## 4.参考资源
1. [Git大全](https://gitee.com/all-about-git)
2. [菜鸟Git教程](https://www.runoob.com/git/git-tutorial.html)
3. [交互式Git命令学习](https://oschina.gitee.io/learn-git-branching/)
4. [Git + GitHub 10分钟完全入门-Bilibili视频教程](https://www.bilibili.com/video/BV1KD4y1S7FL)
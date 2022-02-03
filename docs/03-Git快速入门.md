# 03 - Git 快速入门及在线协同

---

## 1.Git 的安装与 VSCode 集成配置

- Git 下载地址：https://git-scm.com/downloads
- 安装完毕后打开命令提示符（CMD）或终端(Terminal)，输入：`git --version`，若能正常显示 Git 版本，则代表安装成功
- VSCode 会自动与安装的 Git 进行关联，在已有的 Git 项目中，会自动追踪文件的更改，点击 VSCode 界面左侧第三个的 Git 图标将会显示以下界面：

![vscode](https://user-images.githubusercontent.com/1217769/29250412-5b01d194-8042-11e7-90e6-a03e0d78af41.gif)

## 2.Git 基本语法

1. **初始化本地仓库**：用于创建本地新仓库

   1. 首先使用 CMD 或 Terminal 进入项目文件夹 _(使用`cd "path/folder"`命令)_
   2. 输入：**`git init`**
   3. 执行完毕后，目录下将会生成一个名为`.git`的隐藏文件夹，用于 Git 的处理记录，至此该文件夹下的所有文件的修改将被 Git 追踪

2. **克隆远程仓库**：用于下载 Github 等代码托管平台已有的项目到本地
   1. 首先在代码托管平台复制想要克隆的项目 git 地址，Github 平台 git 地址位置如下：
      ![Github](https://docs.github.com/assets/images/help/repository/https-url-clone.png)
      Gitee 平台 git 地址位置如下
      ![Gitee](https://images.gitee.com/uploads/images/2018/0815/115602_7e40b5ff_551147.png "Gitee")
   2. 打开终端进入需要存储的位置，输入：**`git clone https://github.com/xxxxx/xxx.git`**
   3. 命令执行完毕后，该文件夹下会生成与远程仓库一致的所有文件
3. **其他命令**：由于后期主要涉及在 VSCode 上进行 UI 操作，其他命令请参考本文第四节：**参考资源**

## 3.使用 Gitee 进行多人在线协同

### 方法一：贡献者协同（Pull Requests）[💁‍♂️ 推荐]

#### A.轻量级 Pull Requests（最便捷的方法）

Gitee 轻量级 PR（Gitee Pull Request Lite）是一种 无须 Fork 仓库，即可快速向某个特定仓库创建并提交一个合并请求（Pull Request）的功能。步骤如下：

1. 访问本项目 Gitee 的网页，点击进入需要进行更改增减的笔记，在上方工具栏点击`编辑`按钮，直接进行编辑操作

![step1](https://images.gitee.com/uploads/images/2020/0313/082742_a8d773fb_551147.png)

2. 完成编辑后，确保填入“提交信息”（如修改理由等），点击`提交审核`按钮进行 Pull Requests 操作

![step2](https://images.gitee.com/uploads/images/2020/0313/083037_f17b136d_551147.png)

3. 创建成功后，将在**Pull Requests**显示你的提交申请，等待项目作者审核通过即可。

![step3](https://images.gitee.com/uploads/images/2020/0313/083226_55786b86_551147.png)

> 详细操作可以参考 👉[Gitee 官方文档](https://gitee.com/help/articles/4291#article-header2)

#### B.经典 Pull Requests

参与 Gitee 中的仓库开发，最常用和推荐的首选方式是“**Fork + Pull**”模式。在“Fork + Pull”模式下，仓库参与者不必向仓库创建者申请提交权限，而是在自己的托管空间下建立仓库的派生（Fork）。至于在派生仓库中创建的提交，可以非常方便地利用 Gitee 的 Pull Request 工具向原始仓库的维护者发送 Pull Request。

> 详细操作可以参考 👉[Gitee 官方文档](https://gitee.com/help/articles/4128#article-header0)

### 方法二：核心开发组内协同

> 目前鼓励小伙伴们采用贡献者协同方式写作，如有加入核心写作开发组意愿，可向本项目提交`Issues`，写明理由及联系方式进行申请

1. 首先需要每个成员注册各自的 Gitee 账户，步骤参考 👉[此处](https://gitee.com/help/articles/4113)
2. 发起方需要在 Gitee 上创建一个新仓库，步骤参考 👉[此处](https://gitee.com/help/articles/4120)
3. 发起方在仓库管理界面将申请加入协同的其他用户添加至项目的『开发者』，步骤参考 👉[此处](https://gitee.com/help/articles/4175)
4. 所有用户按照本文第二节中的步骤 Clone 新仓库至本地
5. 使用 VSCode 打开本地仓库文件夹，在 VSCode 底部的终端输入以下命令，绑定 Gitee 账户并指定远程仓库地址：

   ```bash
   git config --global user.name "Gitee账户的用户名"
   git config --global user.email "Gitee账户的注册邮箱"
   git remote add origin https://gitee.com/xxxxx/xxx.git
   ```

6. **注意！！！** 在每次协同工作前，为确保本地仓库与远程仓库保持最新的一致性，必须先进行**拉取**(Pull)操作，点击“源代码管理”面板右上角菜单，选择“拉取”

   ![pull](https://cache.yisu.com/upload/information/20210222/263/2395.jpg)

7. 如果在首次拉取时，报错`fatal: refusing to merge unrelated histories`，则在 VSCode 终端中输入：`git pull origin master --allow-unrelated-histories`
8. 接下来便可进行协同工作，在 VSCode 打开的当前仓库文件夹下，创建或修改文件并**保存**，VSCode 集成的 Git 插件会自动跟踪文件修改，并在源代码管理界面列出所有文件更改。
9. 每次修改完在界面左上方的消息框中，输入本次的**Commit**，即修改说明，Commit 指南见 👉[此处](https://gitee.com/help/articles/4231)，填写完毕后点击旁边的 ✔，提交更改，此时 VSCode 将在后台执行一系列命令，自动化完成操作：
   ![commit](https://user-images.githubusercontent.com/1217769/29250412-5b01d194-8042-11e7-90e6-a03e0d78af41.gif)
10. 至此对本地仓库的修改已生效，要使所有成员都能看到你的更改，还需要提交至远程仓库，方法为点击“同步更改”按钮
11. 所有成员将会看到你的更改

## 4.参考资源

1. [Git 大全](https://gitee.com/all-about-git)
2. [菜鸟 Git 教程](https://www.runoob.com/git/git-tutorial.html)
3. [交互式 Git 命令学习](https://oschina.gitee.io/learn-git-branching/)
4. [Git + GitHub 10 分钟完全入门-Bilibili 视频教程](https://www.bilibili.com/video/BV1KD4y1S7FL)

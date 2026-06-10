# Tasks

* [x] Task 1: 设计并实现用户数据模型

  * [x] 1.1 创建用户模型(User)，包含字段：id, username, password\_hash, role, created\_at, updated\_at

  * [x] 1.2 定义角色枚举(Staff, Manager, Admin)

  * [x] 1.3 实现数据库迁移脚本创建用户表

* [x] Task 2: 实现用户认证系统

  * [x] 2.1 实现用户注册API（仅管理员可操作）

  * [x] 2.2 实现用户登录API，返回JWT Token

  * [x] 2.3 实现密码哈希与验证工具函数

  * [x] 2.4 实现JWT Token生成与验证中间件

* [x] Task 3: 实现角色权限控制

  * [x] 3.1 定义各角色的权限规则配置

  * [x] 3.2 实现权限装饰器/中间件

  * [x] 3.3 为现有预订信息接口添加权限控制（staff+）

  * [x] 3.4 为预测相关接口添加权限控制（manager+）

  * [x] 3.5 为管理接口添加权限控制（admin only）
* [x] Task 4: 实现用户管理功能

  * [x] 4.1 实现用户列表查询API（admin only）

  * [x] 4.2 实现用户详情查询API（admin only）

  * [x] 4.3 实现用户创建API（admin only）

  * [x] 4.4 实现用户编辑API（admin only）

  * [x] 4.5 实现用户删除API（admin only）

* [x] Task 5: 集成测试与验证

  * [x] 5.1 测试店员角色权限边界

  * [x] 5.2 测试经理角色权限边界

  * [x] 5.3 测试管理员完整权限

  * [x] 5.4 测试未认证访问拦截

# Task Dependencies

* \[Task 2] depends on \[Task 1]

* \[Task 3] depends on \[Task 2]

* \[Task 4] depends on \[Task 2]

* \[Task 5] depends on \[Task 3, Task 4]


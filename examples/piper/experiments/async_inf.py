import threading
import queue
import time
import random

# ---------- 模拟 VLA 推理（耗时操作） ----------
def policy_model(current_state):
    """
    模拟视觉-语言-动作模型推理。
    根据当前状态返回下一步动作（这里用随机整数模拟）。
    """
    print(f"[推理线程] 开始推理，当前状态 = {current_state}")
    time.sleep(5)  # 模拟 200ms 推理延迟
    action = random.randint(0, 10)
    print(f"[推理线程] 推理完成，生成动作 = {action}")
    return action

# ---------- 模拟动作执行 ----------
def execute_action(action, step_id):
    """
    模拟机器人执行一个动作。
    动作执行耗时 0.1 秒。
    """
    print(f"[执行线程] 第 {step_id} 步：执行动作 {action}")
    time.sleep(1)  # 模拟 100ms 执行耗时
    print(f"[执行线程] 第 {step_id} 步：动作 {action} 执行完毕")

# ---------- 异步推理主程序 ----------
def async_inference_loop(total_steps):
    # 队列用于存放推理好的动作（最多缓存 2 个动作）
    action_queue = queue.Queue(maxsize=2)

    # 当前状态（模拟，这里用 step 编号）
    current_state = 0

    # 启动一个后台线程，持续进行推理
    def inference_worker():
        nonlocal current_state
        while True:
            # 获取当前最新状态（这里简单从主线程获取，实际可用共享变量）
            # 这里我们使用一个全局变量来模拟最新状态，或者从队列中获取？
            # 简化：每次推理前从主线程获取状态，我们用 Event 或直接读取共享变量
            # 这里我们使用一个局部变量 capture_state，通过闭包访问 current_state
            # 注意：为保证数据同步，需要加锁，但示例简化
            state = current_state  # 读主线程状态
            new_action = policy_model(state)
            # 将新动作放入队列（如果队列满，则阻塞等待）
            try:
                action_queue.put(new_action, timeout=0.5)
            except queue.Full:
                print("[推理线程] 队列已满，丢弃旧动作，尝试重新放入")
                # 简单丢弃一个旧动作再放入
                try:
                    action_queue.get_nowait()
                except queue.Empty:
                    pass
                action_queue.put(new_action)

    # 启动推理线程（设为 daemon 以便主程序退出时自动结束）
    inference_thread = threading.Thread(target=inference_worker, daemon=True)
    inference_thread.start()

    # 主循环：执行动作，同时让推理线程在后台不断更新动作队列
    for step in range(total_steps):
        # 如果队列为空，说明推理还没完成，此时主线程应该等待（或执行默认动作）
        # 这里我们等待直到有动作可用
        while action_queue.empty():
            print(f"[主循环] 第 {step} 步：等待推理结果...")
            time.sleep(1)  # 短暂等待

        # 从队列取出一个动作执行
        action = action_queue.get()
        execute_action(action, step)

        # 更新状态（模拟环境反馈）
        current_state = step + 1  # 下一个步骤作为新状态

        # 此时推理线程可能还在继续工作，但我们不用等待，直接进入下一轮循环

    print("所有步骤执行完毕。")

if __name__ == "__main__":
    async_inference_loop(total_steps=8)
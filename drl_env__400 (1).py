import rclpy
from rclpy.node import Node
import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

# --- THE BRAIN (Neural Network) ---
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    def forward(self, x):
        return self.fc(x)

# --- THE LEARNING AGENT ---
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.model = QNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.model(state)
        return torch.argmax(action_values).item()

    def train(self, batch_size=64):
        if len(self.memory) < batch_size: return
        batch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()
            
            output = self.model(state)[action]
            loss = self.criterion(output, torch.tensor(target))
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename):
        torch.save(self.model.state_dict(), filename + ".pth")

# --- THE ENVIRONMENT ---
class DRLEnv(Node):
    def __init__(self):
        super().__init__('drl_env')
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.robot_x, self.robot_y, self.robot_yaw = -2.0, -0.5, 0.0
        self.goal_x, self.goal_y = -1.2, -0.5
        self.ranges = None
        self.prev_distance = 0.0

    def scan_callback(self, msg):
        self.ranges = np.array(msg.ranges)
        self.ranges[np.isinf(self.ranges)] = msg.range_max

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(2*(q.w*q.z + q.x*q.y), 1 - 2*(q.y*q.y + q.z*q.z))

    def get_state(self):
        if self.ranges is None: return np.zeros(26), 0.0
        idx = np.linspace(0, len(self.ranges)-1, 24).astype(int)
        state_lidar = self.ranges[idx] / 3.5
        dx, dy = self.goal_x - self.robot_x, self.goal_y - self.robot_y
        dist = math.sqrt(dx**2 + dy**2)
        angle = (math.atan2(dy, dx) - self.robot_yaw + math.pi) % (2*math.pi) - math.pi
        return np.append(state_lidar, [dist, angle]), dist

    def reset(self):
        self.cmd_pub.publish(Twist())
        self.get_logger().info('🔄 RESET: Manually move robot to (-2, -0.5) and press Ctrl+R!')
        while self.ranges is None: rclpy.spin_once(self, timeout_sec=0.1)
        self.prev_distance = math.sqrt((self.goal_x - (-2.0))**2 + (self.goal_y - (-0.5))**2)
        state, _ = self.get_state()
        return state

    def step(self, action_idx):
        # Map discrete actions to velocities
        actions = [[0.2, 0.0], [0.1, 0.5], [0.1, -0.5], [0.0, 0.5], [0.0, -0.5]]
        move = Twist()
        move.linear.x, move.angular.z = actions[action_idx]
        self.cmd_pub.publish(move)
        
        rclpy.spin_once(self, timeout_sec=0.1)
        state, current_dist = self.get_state()
        reward, done = (self.prev_distance - current_dist) * 250.0, False
        
        if current_dist < 0.22:
            reward, done = 100.0, True
            self.get_logger().info('🎯 SUCCESS!')
        elif np.min(self.ranges) < 0.13:
            reward, done = -100.0, True
            self.get_logger().info('💥 CRASH!')
            
        self.prev_distance = current_dist
        return state, reward, done

# --- THE MAIN LOOP ---
# --- THE MAIN LOOP (FOR THE DEMO) ---
def main(args=None):
    rclpy.init(args=args)
    env = DRLEnv()
    agent = DQNAgent(state_dim=26, action_dim=5)

    # 1. LOAD THE NEW V2 BRAIN
    agent.model.load_state_dict(torch.load("/home/soham/ros2_ws/iiit_dharwad_brain.pth"))
    
    # 2. SET EPSILON TO 0 (Expert Mode - No Randomness)
    agent.epsilon = 0.0 
    
    env.get_logger().info("🚀 EXPERT MODE ACTIVE: Running the v2 Brain...")

    while rclpy.ok():
        state = env.reset()
        for _ in range(500):
            action = agent.select_action(state) # Now uses 100% brain, 0% luck
            next_state, reward, done = env.step(action)
            
            # 3. COMMENT OUT TRAINING (We don't want to change the brain now)
            # agent.memory.append((state, action, reward, next_state, done))
            # agent.train() 
            
            state = next_state
            if done: break

# THIS LINE IS CRITICAL - DO NOT FORGET IT
if __name__ == '__main__':
    main()



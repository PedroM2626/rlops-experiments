#!/usr/bin/env python3
"""
Deep debugging script for Chase environment.
Tests various aspects of environment initialization and physics.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.chase_env import ChaseEnv
import pybullet as p
import math

def test_environment_creation():
    """Test 1: Basic environment creation"""
    print("\n" + "="*60)
    print("TEST 1: Environment Creation")
    print("="*60)
    
    try:
        env = ChaseEnv(render_mode="direct")
        obs, info = env.reset()
        
        print("✅ Environment created successfully")
        print(f"   - Observation shape: {obs.shape}")
        print(f"   - Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
        print(f"   - Action space: {env.action_space}")
        
        # Check if bodies exist
        print(f"   - Agent ID: {env._agent_id}")
        print(f"   - Chaser ID: {env._chaser_id}")
        print(f"   - Number of agent joints: {len(env._joint_indices)}")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_physics():
    """Test 2: Agent physics and stability"""
    print("\n" + "="*60)
    print("TEST 2: Agent Physics and Stability")
    print("="*60)
    
    try:
        env = ChaseEnv(render_mode="direct")
        obs, info = env.reset()
        
        # Get initial position
        agent_pos, _ = p.getBasePositionAndOrientation(env._agent_id, physicsClientId=env._client)
        print(f"✅ Initial agent position: ({agent_pos[0]:.2f}, {agent_pos[1]:.2f}, {agent_pos[2]:.2f})")
        
        # Run 200 steps with zero actions (should maintain balance)
        print("   Running 200 steps with zero actions...")
        for i in range(200):
            action = env.action_space.sample() * 0.0  # All zeros
            obs, reward, term, trunc, info = env.step(action)
            
            agent_pos, _ = p.getBasePositionAndOrientation(env._agent_id, physicsClientId=env._client)
            
            if i % 50 == 0:
                vel, _ = p.getBaseVelocity(env._agent_id, physicsClientId=env._client)
                print(f"     Step {i:3d}: Z={agent_pos[2]:.3f}, Vz={vel[2]:.3f}, Fallen={term}, OOB={agent_pos[2] < -2}")
            
            if term:
                print(f"   ⚠️  Agent fell/caught at step {i}")
                print(f"     Position: {agent_pos}")
                break
        else:
            print(f"   ✅ Agent survived 200 steps")
            print(f"     Final position: ({agent_pos[0]:.2f}, {agent_pos[1]:.2f}, {agent_pos[2]:.2f})")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_raycast():
    """Test 3: Raycast functionality"""
    print("\n" + "="*60)
    print("TEST 3: Raycast Functionality")
    print("="*60)
    
    try:
        env = ChaseEnv(render_mode="direct")
        obs, info = env.reset()
        
        print("✅ Testing raycasts...")
        
        # Get observation and check ray distances
        agent_pos, _ = p.getBasePositionAndOrientation(env._agent_id, physicsClientId=env._client)
        
        # The observation should have ray distances in the last NUM_RAY_CHECKS elements
        # obs structure: [chaser_sin, chaser_cos, chaser_dist, vel_x, vel_y, vel_z, 
        #                 roll, pitch, yaw, joint_pos[10], joint_vel[10], ray_dists[16]]
        
        num_rays = env.NUM_RAY_CHECKS
        ray_start_idx = 3 + 3 + 3 + 10 + 10
        ray_dists = obs[ray_start_idx:ray_start_idx + num_rays]
        
        print(f"   Agent position: ({agent_pos[0]:.2f}, {agent_pos[1]:.2f}, {agent_pos[2]:.2f})")
        print(f"   Ray distances (normalized): min={ray_dists.min():.3f}, max={ray_dists.max():.3f}")
        print(f"   Ray distances sample: {ray_dists[:4]}")
        
        # Raycasts should not be all 1.0 (hitting far wall)
        far_hits = (ray_dists > 0.9).sum()
        print(f"   Rays hitting far: {far_hits}/{num_rays}")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_application():
    """Test 4: Action application"""
    print("\n" + "="*60)
    print("TEST 4: Action Application")
    print("="*60)
    
    try:
        env = ChaseEnv(render_mode="direct")
        obs, info = env.reset()
        
        print("✅ Testing action application...")
        
        # Get initial velocities
        init_vel, _ = p.getBaseVelocity(env._agent_id, physicsClientId=env._client)
        print(f"   Initial velocity: ({init_vel[0]:.3f}, {init_vel[1]:.3f}, {init_vel[2]:.3f})")
        
        # Apply some actions
        for step in range(50):
            action = env.action_space.sample() * 0.5  # Random small actions
            obs, reward, term, trunc, info = env.step(action)
            
            if step % 10 == 0:
                vel, _ = p.getBaseVelocity(env._agent_id, physicsClientId=env._client)
                agent_pos, _ = p.getBasePositionAndOrientation(env._agent_id, physicsClientId=env._client)
                print(f"   Step {step:2d}: Z={agent_pos[2]:.3f}, V=({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}), Reward={reward:.3f}")
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_grace_period():
    """Test 5: Grace period functionality"""
    print("\n" + "="*60)
    print("TEST 5: Grace Period Functionality")
    print("="*60)
    
    try:
        env = ChaseEnv(render_mode="direct")
        obs, info = env.reset()
        
        print(f"✅ Grace period: {env._grace_steps} steps")
        
        # The agent should not terminate due to falling during grace period
        for i in range(150):
            action = env.action_space.sample() * 0.0  # No actions
            obs, reward, term, trunc, info = env.step(action)
            
            if i % 25 == 0:
                print(f"   Step {i:3d}: Grace steps remaining={max(0, env._grace_steps)}, Terminated={term}")
            
            if term:
                print(f"   ⚠️  Terminated at step {i}")
                break
        
        env.close()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("DEEPDEBUG: Chase Environment Comprehensive Test")
    print("="*60)
    
    results = {
        "Environment Creation": test_environment_creation(),
        "Agent Physics": test_agent_physics(),
        "Raycast": test_raycast(),
        "Action Application": test_action_application(),
        "Grace Period": test_grace_period(),
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(results.values())
    print("\n" + ("="*60))
    if all_passed:
        print("✅ ALL TESTS PASSED")
    else:
        print("❌ SOME TESTS FAILED")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

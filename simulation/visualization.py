"""Visualization module for UAV swarm simulation using PyGame."""

import pygame
import numpy as np
import math
import time
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from .uav import UAV
from .disaster_scenario import DisasterScenario
from config.simulation_config import SimulationConfig


@dataclass
class ColorScheme:
    """Color scheme for visualization."""
    background: Tuple[int, int, int] = (240, 240, 240)
    uav: Tuple[int, int, int] = (0, 100, 200)
    uav_trajectory: Tuple[int, int, int] = (100, 150, 255)
    disaster_zone: Tuple[int, int, int] = (255, 100, 100)
    target_area: Tuple[int, int, int] = (100, 255, 100)
    obstacle: Tuple[int, int, int] = (100, 100, 100)
    coverage: Tuple[int, int, int] = (200, 255, 200)
    emergency: Tuple[int, int, int] = (255, 0, 0)
    text: Tuple[int, int, int] = (0, 0, 0)
    grid: Tuple[int, int, int] = (220, 220, 220)


class SwarmVisualizer:
    """PyGame-based visualizer for UAV swarm simulation."""
    
    def __init__(self, config: SimulationConfig, width: int = 1200, height: int = 800):
        self.config = config
        self.width = width
        self.height = height
        self.scale = min(width / config.environment_config.width, 
                        height / config.environment_config.height)
        
        # Initialize PyGame
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MCP-Coordinated Swarm Intelligence")
        
        # Fonts
        self.font_large = pygame.font.Font(None, 24)
        self.font_medium = pygame.font.Font(None, 18)
        self.font_small = pygame.font.Font(None, 14)
        
        # Color scheme
        self.colors = ColorScheme()
        
        # Visualization settings
        self.show_trajectories = True
        self.show_coverage = True
        self.show_communication_links = True
        self.show_info_panel = True
        
        # Camera settings
        self.camera_x = 0
        self.camera_y = 0
        self.zoom = 1.0
        
        # Info panel
        self.info_panel_width = 300
        self.info_panel_height = height
    
    def world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to screen coordinates."""
        screen_x = int((x - self.camera_x) * self.scale * self.zoom + self.width // 2)
        screen_y = int((y - self.camera_y) * self.scale * self.zoom + self.height // 2)
        return screen_x, screen_y
    
    def screen_to_world(self, screen_x: int, screen_y: int) -> Tuple[float, float]:
        """Convert screen coordinates to world coordinates."""
        world_x = (screen_x - self.width // 2) / (self.scale * self.zoom) + self.camera_x
        world_y = (screen_y - self.height // 2) / (self.scale * self.zoom) + self.camera_y
        return world_x, world_y
    
    def draw_background(self):
        """Draw background and grid."""
        self.screen.fill(self.colors.background)
        
        # Draw grid
        if self.zoom > 0.5:
            grid_size = 50 * self.zoom
            # Ensure integer step and bounds for range()
            grid_step = max(1, int(grid_size))
            start_x = int((self.camera_x // grid_size) * grid_size)
            start_y = int((self.camera_y // grid_size) * grid_size)
            end_x = int(self.camera_x + self.width / (self.scale * self.zoom))
            end_y = int(self.camera_y + self.height / (self.scale * self.zoom))
            
            for x in range(start_x, end_x, grid_step):
                screen_x, _ = self.world_to_screen(x, 0)
                if 0 <= screen_x <= self.width:
                    pygame.draw.line(self.screen, self.colors.grid, 
                                   (screen_x, 0), (screen_x, self.height))
            
            for y in range(start_y, end_y, grid_step):
                _, screen_y = self.world_to_screen(0, y)
                if 0 <= screen_y <= self.height:
                    pygame.draw.line(self.screen, self.colors.grid, 
                                   (0, screen_y), (self.width, screen_y))
    
    def draw_disaster_scenario(self, scenario: DisasterScenario):
        """Draw disaster scenario elements."""
        # Draw disaster zones
        for zone in scenario.disaster_zones:
            x, y = self.world_to_screen(zone.x, zone.y)
            
            # Use radius or width/height
            if hasattr(zone, 'radius'):
                r = int(zone.radius * self.scale * self.zoom)
                if -r < x < self.width + r and -r < y < self.height + r:
                    # Circular zone
                    severity_color = (
                        int(255 * zone.severity),
                        int(100 * (1 - zone.severity)),
                        int(100 * (1 - zone.severity))
                    )
                    # Add alpha for pulse if critical
                    if zone.severity > 0.8:
                         pulse = int(10 * np.sin(time.time() * 5))
                         r += pulse
                    
                    pygame.draw.circle(self.screen, severity_color, (int(x), int(y)), int(r))
                    pygame.draw.circle(self.screen, (255, 0, 0), (int(x), int(y)), int(r), 2)
            else:
                w = int(zone.width * self.scale * self.zoom)
                h = int(zone.height * self.scale * self.zoom)
                
                if -w < x < self.width + w and -h < y < self.height + h:
                    # Color based on severity
                    severity_color = (
                        int(255 * zone.severity),
                        int(100 * (1 - zone.severity)),
                        int(100 * (1 - zone.severity))
                    )
                    pygame.draw.rect(self.screen, severity_color, (x, y, w, h))
                    pygame.draw.rect(self.screen, (255, 0, 0), (x, y, w, h), 2)
        
        # Draw obstacles
        for obstacle in scenario.obstacles:
            x, y = self.world_to_screen(obstacle.x, obstacle.y)
            w = int(obstacle.width * self.scale * self.zoom)
            h = int(obstacle.height * self.scale * self.zoom)
            
            if -w < x < self.width + w and -h < y < self.height + h:
                pygame.draw.rect(self.screen, self.colors.obstacle, (x, y, w, h))
                pygame.draw.rect(self.screen, (0, 0, 0), (x, y, w, h), 2)
        
        # Draw target areas
        for target in scenario.target_areas:
            x, y = self.world_to_screen(target.x, target.y)
            w = int(target.width * self.scale * self.zoom)
            h = int(target.height * self.scale * self.zoom)
            
            if -w < x < self.width + w and -h < y < self.height + h:
                # Color based on coverage
                coverage_color = (
                    int(100 + 155 * target.current_coverage),
                    int(255 - 155 * target.current_coverage),
                    100
                )
                pygame.draw.rect(self.screen, coverage_color, (x, y, w, h))
                pygame.draw.rect(self.screen, (0, 255, 0), (x, y, w, h), 2)
        
        # Draw emergency events
        for event in scenario.emergency_events:
            if not event.resolved:
                x, y = self.world_to_screen(event.x, event.y)
                radius = int(10 * self.scale * self.zoom)
                
                if -radius < x < self.width + radius and -radius < y < self.height + radius:
                    pygame.draw.circle(self.screen, self.colors.emergency, (x, y), radius)
                    pygame.draw.circle(self.screen, (255, 255, 0), (x, y), radius, 2)
    
    def draw_uav(self, uav: UAV, index: int = 0):
        """Draw a single UAV."""
        x, y = self.world_to_screen(uav.state.x, uav.state.y)
        
        if -20 < x < self.width + 20 and -20 < y < self.height + 20:
            # Draw UAV body
            color = self.colors.uav
            if uav.state.status == "low_battery":
                color = (255, 165, 0)  # Orange
            elif uav.state.status == "emergency":
                color = (255, 0, 0)  # Red
            
            pygame.draw.circle(self.screen, color, (x, y), 8)
            pygame.draw.circle(self.screen, (0, 0, 0), (x, y), 8, 2)
            
            # Draw velocity vector
            if uav.state.vx != 0 or uav.state.vy != 0:
                vx, vy = self.world_to_screen(
                    uav.state.x + uav.state.vx * 2,
                    uav.state.y + uav.state.vy * 2
                )
                pygame.draw.line(self.screen, (0, 0, 0), (x, y), (vx, vy), 2)
            
            # Draw sensor range
            if self.zoom > 0.3:
                sensor_radius = int(uav.sensor_range * self.scale * self.zoom)
                pygame.draw.circle(self.screen, (200, 200, 255), (x, y), sensor_radius, 1)
            
            # Draw UAV ID
            if self.zoom > 0.5:
                text = self.font_small.render(f"UAV-{index}", True, self.colors.text)
                self.screen.blit(text, (x + 10, y - 10))
    
    def draw_uav_trajectory(self, uav: UAV):
        """Draw UAV trajectory."""
        if not self.show_trajectories or len(uav.trajectory) < 2:
            return
        
        points = []
        for pos in uav.trajectory[-50:]:  # Last 50 positions
            x, y = self.world_to_screen(pos[0], pos[1])
            if 0 <= x <= self.width and 0 <= y <= self.height:
                points.append((x, y))
        
        if len(points) > 1:
            pygame.draw.lines(self.screen, self.colors.uav_trajectory, False, points, 1)
    
    def draw_communication_links(self, uavs: List[UAV]):
        """Draw communication links between UAVs."""
        if not self.show_communication_links:
            return
        
        for i, uav1 in enumerate(uavs):
            for j, uav2 in enumerate(uavs[i+1:], i+1):
                if uav1.can_communicate_with(uav2):
                    x1, y1 = self.world_to_screen(uav1.state.x, uav1.state.y)
                    x2, y2 = self.world_to_screen(uav2.state.x, uav2.state.y)
                    
                    if (0 <= x1 <= self.width and 0 <= y1 <= self.height and
                        0 <= x2 <= self.width and 0 <= y2 <= self.height):
                        pygame.draw.line(self.screen, (100, 100, 100), (x1, y1), (x2, y2), 1)
    
    def draw_coverage(self, scenario: DisasterScenario):
        """Draw coverage grid."""
        if not self.show_coverage:
            return
        
        grid = scenario.coverage_grid
        cell_size = 10 * self.scale * self.zoom
        
        for y in range(grid.shape[0]):
            for x in range(grid.shape[1]):
                if grid[y, x] > 0:
                    world_x = x * 10
                    world_y = y * 10
                    screen_x, screen_y = self.world_to_screen(world_x, world_y)
                    
                    if -cell_size < screen_x < self.width + cell_size and -cell_size < screen_y < self.height + cell_size:
                        alpha = int(100 * grid[y, x])
                        color = (*self.colors.coverage, alpha)
                        # Note: PyGame doesn't support alpha in draw.rect directly
                        # This is a simplified version
                        pygame.draw.rect(self.screen, self.colors.coverage, 
                                       (screen_x, screen_y, int(cell_size), int(cell_size)))
    
    def draw_info_panel(self, uavs: List[UAV], scenario: DisasterScenario, 
                       simulation_time: float, fps: float):
        """Draw information panel."""
        if not self.show_info_panel:
            return
        
        panel_x = self.width - self.info_panel_width
        panel_y = 0
        
        # Draw panel background
        pygame.draw.rect(self.screen, (250, 250, 250), 
                        (panel_x, panel_y, self.info_panel_width, self.info_panel_height))
        pygame.draw.rect(self.screen, (200, 200, 200), 
                        (panel_x, panel_y, self.info_panel_width, self.info_panel_height), 2)
        
        # Draw info text
        y_offset = 20
        line_height = 20
        
        # Simulation info
        info_texts = [
            f"Simulation Time: {simulation_time:.1f}s",
            f"FPS: {fps:.1f}",
            f"UAVs: {len(uavs)}",
            f"Coverage: {scenario.get_coverage_percentage():.1f}%",
            "",
            "UAV Status:"
        ]
        
        for text in info_texts:
            if text:
                rendered_text = self.font_medium.render(text, True, self.colors.text)
                self.screen.blit(rendered_text, (panel_x + 10, y_offset))
            y_offset += line_height
        
        # UAV status
        for i, uav in enumerate(uavs):
            status_text = f"UAV-{i}: {uav.state.status}"
            battery_text = f"  Battery: {uav.state.battery:.1f}%"
            pos_text = f"  Pos: ({uav.state.x:.0f}, {uav.state.y:.0f})"
            
            for text in [status_text, battery_text, pos_text]:
                rendered_text = self.font_small.render(text, True, self.colors.text)
                self.screen.blit(rendered_text, (panel_x + 10, y_offset))
                y_offset += 15
        
        # Scenario info
        y_offset += 10
        scenario_texts = [
            "Scenario:",
            f"Disaster Zones: {len(scenario.disaster_zones)}",
            f"Obstacles: {len(scenario.obstacles)}",
            f"Target Areas: {len(scenario.target_areas)}",
            f"Emergency Events: {len([e for e in scenario.emergency_events if not e.resolved])}"
        ]
        
        for text in scenario_texts:
            rendered_text = self.font_medium.render(text, True, self.colors.text)
            self.screen.blit(rendered_text, (panel_x + 10, y_offset))
            y_offset += line_height
    
    def handle_events(self):
        """Handle PyGame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_t:
                    self.show_trajectories = not self.show_trajectories
                elif event.key == pygame.K_c:
                    self.show_coverage = not self.show_coverage
                elif event.key == pygame.K_l:
                    self.show_communication_links = not self.show_communication_links
                elif event.key == pygame.K_i:
                    self.show_info_panel = not self.show_info_panel
                elif event.key == pygame.K_r:
                    self.camera_x = 0
                    self.camera_y = 0
                    self.zoom = 1.0
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Mouse wheel up
                    self.zoom = min(3.0, self.zoom * 1.1)
                elif event.button == 5:  # Mouse wheel down
                    self.zoom = max(0.1, self.zoom / 1.1)
        
        # Handle camera movement with arrow keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.camera_x -= 10 / self.zoom
        if keys[pygame.K_RIGHT]:
            self.camera_x += 10 / self.zoom
        if keys[pygame.K_UP]:
            self.camera_y -= 10 / self.zoom
        if keys[pygame.K_DOWN]:
            self.camera_y += 10 / self.zoom
        
        return True
    
    def render(self, uavs: List[UAV], scenario: DisasterScenario, 
               simulation_time: float, fps: float):
        """Render the complete simulation."""
        # Handle events
        if not self.handle_events():
            return False
        
        # Draw everything
        self.draw_background()
        self.draw_disaster_scenario(scenario)
        
        if self.show_coverage:
            self.draw_coverage(scenario)
        
        # Draw UAVs and their trajectories
        for i, uav in enumerate(uavs):
            self.draw_uav_trajectory(uav)
            self.draw_uav(uav, i)
        
        if self.show_communication_links:
            self.draw_communication_links(uavs)
        
        if self.show_info_panel:
            self.draw_info_panel(uavs, scenario, simulation_time, fps)
        
        # Update display
        pygame.display.flip()
        
        return True
    
    def cleanup(self):
        """Clean up PyGame resources."""
        pygame.quit()

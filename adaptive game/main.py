import pygame
import sys
import csv
import time

# -------------------------
# CONFIGURATION (FIXED)
# -------------------------
TILE_SIZE = 32
GRID_WIDTH = 10
GRID_HEIGHT = 10
FPS = 60
print("Setting up the game environment...")

WINDOW_WIDTH = GRID_WIDTH * TILE_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * TILE_SIZE

# -------------------------
# WORLD (1 = wall, 0 = floor)
# -------------------------
WORLD_MAP = [
    [1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,1],
    [1,1,1,1,1,1,1,1,1,1],
]

# -------------------------
# INITIALIZE PYGAME
# -------------------------
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Phase 1: Controlled Environment")
clock = pygame.time.Clock()

# -------------------------
# COLORS
# -------------------------
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (50, 100, 255)
RED = (200, 50, 50)
GRAY = (180, 180, 180)

# -------------------------
# PLAYER & ENEMY STATE
# -------------------------
player_pos = [1, 1]
enemy_pos = [8, 8]

player_health = 100
enemy_health = 100

# -------------------------
# LOGGER SETUP
# -------------------------
log_file = open("phase1_log.csv", "w", newline="")
logger = csv.writer(log_file)
logger.writerow([
    "time",
    "player_x", "player_y",
    "enemy_x", "enemy_y",
    "player_health", "enemy_health"
])

start_time = time.time()

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def can_move(x, y):
    return WORLD_MAP[y][x] == 0

def draw_world():
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(
                x * TILE_SIZE,
                y * TILE_SIZE,
                TILE_SIZE,
                TILE_SIZE
            )
            if WORLD_MAP[y][x] == 1:
                pygame.draw.rect(screen, GRAY, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)

def move_enemy_towards_player():
    # Stop if already adjacent (distance 1)
    if manhattan_distance(player_pos, enemy_pos) <= 1:
        return

    dx = player_pos[0] - enemy_pos[0]
    dy = player_pos[1] - enemy_pos[1]

    if abs(dx) > abs(dy):
        step_x = 1 if dx > 0 else -1
        new_x = enemy_pos[0] + step_x
        if can_move(new_x, enemy_pos[1]):
            enemy_pos[0] = new_x
    else:
        step_y = 1 if dy > 0 else -1
        new_y = enemy_pos[1] + step_y
        if can_move(enemy_pos[0], new_y):
            enemy_pos[1] = new_y


def manhattan_distance(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# -------------------------
# MAIN GAME LOOP
# -------------------------
while True:
    clock.tick(FPS)
        
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            log_file.close()
            pygame.quit()
            sys.exit()

        if event.type == pygame.QUIT:
            log_file.close()
            pygame.quit()
            sys.exit()

        if event.type == pygame.KEYDOWN:
            print("pressed")
            if event.key == pygame.K_UP:
                if can_move(player_pos[0], player_pos[1] - 1):
                    player_pos[1] -= 1
            elif event.key == pygame.K_DOWN:
                if can_move(player_pos[0], player_pos[1] + 1):
                    player_pos[1] += 1
            elif event.key == pygame.K_LEFT:
                if can_move(player_pos[0] - 1, player_pos[1]):
                    player_pos[0] -= 1
            elif event.key == pygame.K_RIGHT:
                if can_move(player_pos[0] + 1, player_pos[1]):
                    player_pos[0] += 1

    # Enemy movement (simple chase)
    move_enemy_towards_player()

    # Logging
    current_time = time.time() - start_time
    logger.writerow([
        round(current_time, 2),
        player_pos[0], player_pos[1],
        enemy_pos[0], enemy_pos[1],
        player_health, enemy_health
    ])

    # Rendering
    screen.fill(BLACK)
    draw_world()

    pygame.draw.rect(
        screen,
        BLUE,
        pygame.Rect(
            player_pos[0] * TILE_SIZE,
            player_pos[1] * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE
        )
    )

    pygame.draw.rect(
        screen,
        RED,
        pygame.Rect(
            enemy_pos[0] * TILE_SIZE,
            enemy_pos[1] * TILE_SIZE,
            TILE_SIZE,
            TILE_SIZE
        )
    )

    pygame.display.flip()

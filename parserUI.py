import pygame
import requests
import time
import json

# Initialize pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("GNSS Data Viewer")

# Set up fonts
title_font = pygame.font.Font(None, 48)
header_font = pygame.font.Font(None, 36)
text_font = pygame.font.Font(None, 28)

# Colors
bg_color = (30, 30, 30)
title_color = (255, 255, 255)
header_color = (100, 200, 255)
text_color = (200, 200, 200)

# Function to fetch data from the server
def fetch_data():
    try:
        response = requests.get('http://127.0.0.1:2121/latest_data')
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Function to render text with word wrap
def render_textrect(string, font, rect, text_color, bg_color, justification=0):
    final_lines = []
    requested_lines = string.splitlines()

    # Create a series of lines that will fit in the provided rectangle.
    for requested_line in requested_lines:
        if font.size(requested_line)[0] > rect.width:
            words = requested_line.split(' ')
            accumulated_line = ""
            for word in words:
                test_line = accumulated_line + word + " "
                if font.size(test_line)[0] < rect.width:
                    accumulated_line = test_line
                else:
                    final_lines.append(accumulated_line)
                    accumulated_line = word + " "
            final_lines.append(accumulated_line)
        else:
            final_lines.append(requested_line)

    # Let's try to write the text out on the surface.
    surface = pygame.Surface(rect.size)
    surface.fill(bg_color)
    accumulated_height = 0
    for line in final_lines:
        if accumulated_height + font.size(line)[1] >= rect.height:
            break
        if line != "":
            tempsurface = font.render(line, 1, text_color)
            if justification == 0:
                surface.blit(tempsurface, (0, accumulated_height))
            elif justification == 1:
                surface.blit(tempsurface, ((rect.width - tempsurface.get_width()) / 2, accumulated_height))
            elif justification == 2:
                surface.blit(tempsurface, (rect.width - tempsurface.get_width(), accumulated_height))
            accumulated_height += font.size(line)[1]
    return surface

# Main loop
running = True
last_measurement = None
last_position = None

# Scroll variables
scroll_y = 0
scroll_speed = 20

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 4:  # Scroll up
                scroll_y = max(scroll_y - scroll_speed, 0)
            elif event.button == 5:  # Scroll down
                scroll_y += scroll_speed

    # Fetch the latest data
    data = fetch_data()
    if data:
        last_measurement = data.get("measurement")
        last_position = data.get("position")

    # Clear the screen
    screen.fill(bg_color)

    # Display the title
    title_text = title_font.render("GNSS Data Viewer", True, title_color)
    screen.blit(title_text, (width // 2 - title_text.get_width() // 2, 20))

    # Display the last received measurement
    measurement_header = header_font.render("Last Received Measurement:", True, header_color)
    screen.blit(measurement_header, (20, 80))

    if last_measurement:
        measurement_text = json.dumps(last_measurement, indent=2)
        rect = pygame.Rect(20, 120, width - 40, height // 2 - 120)
        rendered_text = render_textrect(measurement_text, text_font, rect, text_color, bg_color)
        screen.blit(rendered_text, (20, 120 - scroll_y))
    else:
        no_data_text = text_font.render("No data received", True, text_color)
        screen.blit(no_data_text, (20, 120))

    # Display the last calculated position
    position_header = header_font.render("Last Calculated Position:", True, header_color)
    screen.blit(position_header, (20, height // 2))

    if last_position:
        position_details = ["Latitude: {:.6f}".format(last_position[0]),
                            "Longitude: {:.6f}".format(last_position[1]),
                            "Altitude: {:.2f} meters".format(last_position[2])]
        y_offset = height // 2 + 40
        for detail in position_details:
            text = text_font.render(detail, True, text_color)
            screen.blit(text, (20, y_offset))
            y_offset += 30
    else:
        no_position_text = text_font.render("No position calculated", True, text_color)
        screen.blit(no_position_text, (20, height // 2 + 40))

    # Update the display
    pygame.display.flip()

    # Wait before fetching data again
    time.sleep(10)

pygame.quit()

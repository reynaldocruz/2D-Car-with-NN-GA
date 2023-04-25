import pygame

def scale_image(img, factor):
    size = round(img.get_width() * factor), round(img.get_height() * factor)
    return pygame.transform.scale(img, size)

def blit_rotate_center(win, image, top_left, angle):
    rotate_image = pygame.transform.rotate(image, angle)
    new_rect = rotate_image.get_rect(center=image.get_rect(topleft=top_left).center)
    win.blit(rotate_image, new_rect.topleft)

def blit_text_center(win, font, text):
    render = font.render(text, 1, (200,200,200))
    win.blit(render, (win.get_width()/2 - render.get_width() /
                      2, win.get_height()/2 - render.get_height()/2))
    
def contains(list, filter):
    for x in list:
        if filter(x):
            return True
    return False
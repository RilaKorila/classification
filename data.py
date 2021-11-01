from PIL import Image

def my_dtree(feature1, feature2):
    f_name = './tree_imgs/' + feature1 + '_' + feature2 +  '.png'
    image = Image.open(f_name)
    
    return image
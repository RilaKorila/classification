from PIL import Image

def my_dtree(feature1, feature2):
    # if feature1 == 'Gender':
    #     feature1 = 'Sex'
    # if feature2 == 'Gender':
    #     feature2 = 'Sex'
    # f_name = './tree_imgs_png/' + min(feature1, feature2)  + '_' + max(feature1, feature2) +  '.png'
    tmp_path = "tree_imgs/PetalWidthCm_SepalLengthCm.png"
    image = Image.open(tmp_path)
    
    return image
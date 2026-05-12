import numpy as np

"kol ma sigma btzeed el noise btzeed"
def add_gaussian_noise(image, sigma=25):
    h, w = image.shape

   
    "b3ml 2grids mn random numbers for each pixel, 3shan a2dr a3ml el transformation bta3 box-muller method"
    "var,mean,image size"
    u1 = np.random.uniform(0.0001, 1, (h, w))
    u2 = np.random.uniform(0.0001, 1, (h, w))

    "box-muller method"
    z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)

    noise = sigma * z

    "related to g(x,y) = f(x,y) + n(x,y) , noise 7aga decimal f kan lazm a8yr el image le decimals "
    noisy = image.astype(np.float64) + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)

" bady range llnoise " 
def add_uniform_noise(image, low=50, high=50):
    h, w = image.shape
    "b3ml grid wa7d,values ely f range 3ndha nfs elprobability"
    noise = np.random.uniform(-low, high, (h, w))

    noisy = image.astype(np.float64) + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)
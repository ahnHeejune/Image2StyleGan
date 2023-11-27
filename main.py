from image2stylegan import G_mapping,G_synthesis
from VGG16 import VGG16_perceptual
from utils import image_reader, loss_function, PSNR, get_device
import torch
import torch.optim as optim
import torch.nn as nn
from collections import OrderedDict
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import argparse
import os



# generate psuedo mean w using random z vectors 
def make_mean_latent_W():

    n_sample = 1000
    mean_w = None

    for i in range(n_sample):
        z_latent = 2 * torch.randn((100, 1, 512), device=device) - 1
        w_latent = g_mapping(z_latent)
        mean = torch.mean(w_latent, dim=0, keepdim=True)
        if mean_w == None:
            mean_w = mean
        else:
            mean_w += mean
    mean_w /= n_sample 
    print(mean_w.shape)
    #print(mean_w)

    return mean_w[0,0:1,:]

def embedding_to_W(image, w_init = None, file_name = None):

    #encode images to W space (512)

    perceptual = VGG16_perceptual().to(device)
    MSE_loss = nn.MSELoss(reduction="mean")
    if w_init is None:
        latent_w = torch.zeros((1, 512), requires_grad=True, device=device)
    else:
        init_value = w_init.detach().cpu().numpy().tolist()
        latent_w = torch.tensor(init_value, requires_grad=True, device=device)

    optimizer = optim.Adam({latent_w}, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    viz = False
    if viz:
        loss_ = []
        loss_psnr = []
    upsample = torch.nn.Upsample(scale_factor=256 / 1024, mode='bilinear')
    img_p = image.clone()
    img_p = upsample(img_p)

    for e in range(args.epochs):
        optimizer.zero_grad()
        latent_w1 = latent_w.unsqueeze(1).expand(-1, 18, -1)
        syn_img = g_synthesis(latent_w1)
        syn_img = (syn_img + 1.0) / 2.0
        mse, per_loss = loss_function(syn_img, image, img_p, MSE_loss, upsample, perceptual)
        psnr = PSNR(mse, flag=0)
        loss = per_loss + mse
        loss.backward()
        optimizer.step()
        loss_np = loss.detach().cpu().numpy()
        loss_p = per_loss.detach().cpu().numpy()
        loss_m = mse.detach().cpu().numpy()
        if viz:
            loss_psnr.append(psnr)
            loss_.append(loss_np)
        if (e + 1) % 500 == 0:
            print("iter{}: loss: {:.2f} = {:.2f} (mse) + {:.2f} (percep), psnr:{:.2f}".format(e + 1, loss_np, loss_m, loss_p, psnr))
            if file_name is not None:
                file_path =  "save_images/image2stylegan/explore/W_space/reconstruction/reconstruct-{}-{}.png".format(file_name, e + 1)
                save_image(syn_img.clamp(0, 1), file_path)

    if viz:
        plt.plot(loss_, label='Loss = MSELoss + Perceptual')
        plt.plot(loss_psnr, label='PSNR')
        plt.legend()

    #print(loss_psnr[-1])
    return latent_w, syn_img


def embedding_to_Z(image, image_id):
    '''
        encode images to Z space
    '''
    upsample = torch.nn.Upsample(scale_factor=256 / 1024, mode='bilinear')
    img_p = image.clone()
    img_p = upsample(img_p)

    perceptual = VGG16_perceptual().to(device)
    MSE_loss = nn.MSELoss(reduction="mean")
    latents = torch.zeros(1, 512, requires_grad=True, device=device)
    optimizer = optim.Adam({latents}, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    loss_ = []
    loss_psnr = []
    for e in range(args.epochs):

        optimizer.zero_grad()
        syn_img = g_all(latents)   # mpping and synthesis
        syn_img = (syn_img + 1.0) / 2.0
        mse, per_loss = loss_function(syn_img, image, img_p, MSE_loss, upsample, perceptual)
        psnr = PSNR(mse, flag=0)
        loss = per_loss + mse
        loss.backward()
        optimizer.step()

        loss_np = loss.detach().cpu().numpy()
        loss_p = per_loss.detach().cpu().numpy()
        loss_m = mse.detach().cpu().numpy()
        loss_psnr.append(psnr)
        loss_.append(loss_np)
        if (e + 1) % 500 == 0:
            print("iter{}: loss -- {},  mse_loss --{},  percep_loss --{}, psnr --{}".format(e + 1, loss_np, loss_m,
                                                                                            loss_p, psnr))
            save_image(syn_img.clamp(0, 1),
                       "save_images/image2stylegan/explore/Z_space/reconstruction/reconstruct_{}_{}.png".format(image_id, e + 1))

    plt.plot(loss_, label='Loss = MSELoss + Perceptual')
    plt.plot(loss_psnr, label='PSNR')
    plt.legend()
    print(loss_psnr[-1])
    return latents, syn_img


def embedding_to_Wplus(image, file_name = None):
    '''
    image to W_latent 
    '''

    perceptual_loss = VGG16_perceptual().to(device)
    MSE_loss = nn.MSELoss(reduction="mean")
    latents = torch.zeros((1, 18, 512), requires_grad=True, device=device)
    optimizer = optim.Adam({latents}, lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    viz = False
    if viz:
        loss_ = []
        loss_psnr = []
    upsample = torch.nn.Upsample(scale_factor=256 / 1024, mode='bilinear')
    img_p = image.clone()
    img_p = upsample(img_p)
    for e in range(args.epochs):

        # forward and optimization 
        optimizer.zero_grad()
        syn_img = g_synthesis(latents)
        syn_img = (syn_img + 1.0) / 2.0
        mse, per_loss = loss_function(syn_img, image, img_p, MSE_loss, upsample, perceptual_loss)
        psnr = PSNR(mse, flag=0)
        loss = per_loss + mse
        loss.backward()
        optimizer.step()

        # result 
        loss_np = loss.detach().cpu().numpy()
        loss_p = per_loss.detach().cpu().numpy()
        loss_m = mse.detach().cpu().numpy()
        if viz:
            loss_psnr.append(psnr)
            loss_.append(loss_np)
        if (e + 1) % 500 == 0:
            print("iter{}: loss={:.2f}={:.2f}(mse)+{:.2f}(percep), psnr={:.2f}".format(e + 1, loss_np, loss_m, loss_p, psnr))
            if file_name is not None:
                save_image(syn_img.clamp(0, 1), "save_images/image2stylegan/reconstruction/{}_{}.png".format(file_name, e + 1))
                #save_image(syn_img.clamp(0, 1), "save_images/image2stylegan/reconstruction/reconstruct_{}.png".format(e + 1))

    if viz:
        plt.plot(loss_, label='Loss = MSELoss + Perceptual')
        plt.plot(loss_psnr, label='PSNR')
        plt.legend()
    return latents, syn_img


def morphing_in_Wplus(wplus0, wplus1, alpha = 0.5):
    '''
        morphing operation
    '''
    wplus = wplus0 * (1 - alpha) + wplus1 * alpha   # alpha blending the w_latent vector 
    syn_img = g_synthesis(wplus)    # generate image value in (-1.0, +1.0) 
    syn_img = (syn_img + 1.0) / 2.0   #  (-1,+1) to (0, 1)
    return  wplus, syn_img 

def morphing_in_W(w0, w1, alpha = 0.5):
    '''
        morphing operation
    '''
    w = w0 * (1 - alpha) + w1 * alpha   # alpha blending the w_latent vector 
    wplus = w.unsqueeze(1).expand(-1, 18, -1)
    syn_img = g_synthesis(wplus)    # generate image value in (-1.0, +1.0) 
    syn_img = (syn_img + 1.0) / 2.0   #  (-1,+1) to (0, 1)
    return  w, syn_img 

def demorph_in_Wplus(wplus_morphed, wplus_live, alpha = 0.5):
    '''
        demorphing operation
    '''
    wplus = 2.*(wplus_morphed - alpha * wplus_live)  # alpha blending the w_latent vector 
    syn_img = g_synthesis(wplus)    # generate image value in (-1.0, +1.0) 
    syn_img = (syn_img + 1.0) / 2.0   #  (-1,+1) to (0, 1)
    return syn_img

def demorph_in_W(w_morphed, w_live, alpha = 0.5):
    '''
        demorphing operation
    '''
    w = 2.*(w_morphed - alpha * w_live)  # alpha blending the w_latent vector 
    wplus = w.unsqueeze(1).expand(-1, 18, -1)
    syn_img = g_synthesis(wplus)    # generate image value in (-1.0, +1.0) 
    syn_img = (syn_img + 1.0) / 2.0   #  (-1,+1) to (0, 1)
    return syn_img


def style_transfer(target_latent, style_latent, src, tgt):
    '''
        style transfer
    '''
    tmp_latent1 = target_latent[:, :10, :]   # large scale w_latent (contents) 
    tmp_latent2 = style_latent[:, 10:, :]    # small scale w_latent (style)
    latent = torch.cat((tmp_latent1, tmp_latent2), dim=1)
    #print(latent.shape)
    print(latent.size())
    syn_img = g_synthesis(latent)
    syn_img = (syn_img + 1.0) / 2.0
    save_image(syn_img.clamp(0, 1), "save_images/image2stylegan/style_transfer/Style_transfer_{}_{}_10.png".format(src, tgt))


def check_repeatibility():

    torch.manual_seed(0)

    img_path = os.path.join(args.images_dir, "0.png")
    image0 = image_reader(img_path)
    image0 = image0.to(device)

    w_latent0, syn_img = embedding_to_W(image0, mean_W, file_name = "0")
    print("First w[::16]:", w_latent0.detach().cpu().squeeze().numpy()[::16])

    w_latent0, syn_img = embedding_to_W(image0, mean_W, file_name = "0")
    print("second w[::16]:", w_latent0.detach().cpu().squeeze().numpy()[::16])




def image2StyleGan_W():

    #  w plus space 
    #  512x8 independent latent variable 

    # load test images
    print("loading images and embedding 0.")
    img_path = os.path.join(args.images_dir, "0.png")
    image0 = image_reader(img_path)
    image0 = image0.to(device)
    w_latent0, syn_img = embedding_to_W(image0, file_name = "0")


    '''
    init_value = w_latent0.detach().cpu().numpy().tolist()
    latent_w = torch.tensor(init_value, requires_grad=True, device=device)
    print(f"{latent_w}")
    '''
    

    img_path = os.path.join(args.images_dir, "4.png")
    image4 = image_reader(img_path)
    image4 = image4.to(device)
    w_latent4 , syn_img= embedding_to_W(image4, file_name = "4")
    if False:
        plt.subplot(1,2,1)
        plt.imshow(image4.detach().cpu().squeeze().permute((1,2,0)).numpy())
        plt.title("input")
        plt.subplot(1,2,2)
        plt.imshow(syn_img.detach().cpu().squeeze().permute((1,2,0)).numpy())
        plt.title("reconst.")
        plt.show()

    print("loading images and embedding 5.")
    img_path = os.path.join(args.images_dir, "5.png")
    image5 = image_reader(img_path)
    image5 = image5.to(device)
    w_latent5, syn_img = embedding_to_W(image5, file_name = "5")
    if False:
        plt.subplot(1,2,1)
        plt.imshow(image5.detach().cpu().squeeze().permute((1,2,0)).numpy())
        plt.title("input")
        plt.subplot(1,2,2)
        plt.imshow(syn_img.detach().cpu().squeeze().permute((1,2,0)).numpy())
        plt.title("reconst.")
        plt.show()

    print("loading images and embedding 7.")
    img_path = os.path.join(args.images_dir, "7.png")
    image7 = image_reader(img_path)
    image7 = image7.to(device)
    w_latent7, syn_img = embedding_to_W(image7, file_name = "7")

    print(f"size of w latent: {w_latent7.size()}")

    # morphing test 
    print("morping starts")
    nsteps = 10
    for i in range(nsteps+1):
        alpha = i/nsteps
        _, syn_img = morphing_in_W(w_latent4, w_latent5, alpha)
        file_path =  "save_images/image2stylegan/explore/W_space/morphing/morphed_{}_{}_{}.png".format(4, 5, int(alpha*10))
        save_image(syn_img.clamp(0, 1), file_path) 
    alpha = 0.5
    w_latent_morphed0, syn_img = morphing_in_W(w_latent4, w_latent5, 0.5)
    morphed_path =  "save_images/image2stylegan/explore/W_space/morphing/morphed_{}_{}_{}.png".format(4, 5, int(alpha*10))
    save_image(syn_img.clamp(0, 1), morphed_path) 
    print(f"size of w latent: {w_latent_morphed0.size()}")
    print("original morphed w[::16]:", w_latent_morphed0.detach().cpu().squeeze().numpy()[::16])
    print("morping done")
    
    print("demorping starts")
    use_meanW = True
    print("loading and embedding the moprphed image")
    image_morphed = image_reader(morphed_path)
    image_morphed = image_morphed.to(device)
    if use_meanW:
        w_latent_morphed1, _ = embedding_to_W(image_morphed)
    else:
        w_latent_morphed1, _ = embedding_to_W(image_morphed, w_latent5)

    print("recon morphed w[::16]:", w_latent_morphed1.detach().cpu().squeeze().numpy()[::16])
    w_diff = w_latent_morphed1 - w_latent_morphed0
    print("diff morphed w[::16]:", w_diff.detach().cpu().squeeze().numpy()[::16])

    print("loading and embedding the live image")
    img_path = os.path.join(args.images_dir, "5.png")
    image_live = image_reader(img_path)
    image_live = image_live.to(device)
    if use_meanW:
        w_latent_live, _  = embedding_to_W(image_live, w_latent5)  # not mean but any face
    else:
        w_latent_live, _  = embedding_to_W(image_live)

    w_diff_live = w_latent_live - w_latent5
    print("diff live w[::16]:", w_diff_live.detach().cpu().squeeze().numpy()[::16])
    print("live w[::16]:", w_latent_live.detach().cpu().squeeze().numpy()[::16])
    print("5  w[::16]:", w_latent5.detach().cpu().squeeze().numpy()[::16])

    plt.subplot(2,4,1)
    plt.imshow(image4.detach().cpu().squeeze().permute((1,2,0)).numpy())
    plt.subplot(2,4,2)
    plt.imshow(image_morphed.detach().cpu().squeeze().permute((1,2,0)).numpy())
    plt.title("morphed")
    plt.subplot(2,4,3)
    plt.imshow(image5.detach().cpu().squeeze().permute((1,2,0)).numpy())

    print("demorphing")
    syn_img = demorph_in_W(w_latent_morphed0, w_latent5)
    plt.subplot(2,4,5)
    plt.imshow(syn_img.detach().cpu().squeeze().permute((1,2,0)).numpy())
    file_path =  "save_images/image2stylegan/explore/W_space/morphing/{}.png".format("demorph0")
    save_image(syn_img.clamp(0, 1), file_path)
    syn_img = demorph_in_W(w_latent_morphed1, w_latent_live)  ### important 
    plt.subplot(2,4,6)
    plt.imshow(syn_img.detach().cpu().squeeze().permute((1,2,0)).numpy())
    file_path =  "save_images/image2stylegan/explore/W_space/morphing/{}.png".format("demorph1")
    save_image(syn_img.clamp(0, 1), file_path)
    syn_img = demorph_in_W(w_latent_morphed0, w_latent_live)  
    plt.subplot(2,4,7)
    plt.imshow(syn_img.detach().cpu().squeeze().permute((1,2,0)).numpy())
    file_path =  "save_images/image2stylegan/explore/W_space/morphing/{}.png".format("demorph2")
    save_image(syn_img.clamp(0, 1), file_path)
    syn_img = demorph_in_W(w_latent_morphed1, w_latent5) 
    plt.subplot(2,4,8)
    plt.imshow(syn_img.detach().cpu().squeeze().permute((1,2,0)).numpy())
    file_path =  "save_images/image2stylegan/explore/W_space/morphing/{}.png".format("demorph3")
    save_image(syn_img.clamp(0, 1), file_path)
    print("demorphing done")
    plt.show()


    ##############################################################
    # demopring =  morphing - 4
    ##############################################################

    ##############################################################
    # demopring =  5 - 5
    ##############################################################

    ##############################################################
    # demopring =  4 - 4
    ##############################################################

    print("demorphing done")

    # style transfer 
    wplus_latent0 = w_latent0.unsqueeze(1).expand(-1, 18, -1)
    wplus_latent7 = w_latent7.unsqueeze(1).expand(-1, 18, -1)
    print("style transfer starts")
    style_transfer(wplus_latent0, wplus_latent7, 0, 7)
    print("style transfer done")
   
    print("Success in all")



def image2StyleGan_Wplus():

    #  w plus space 
    #  512x8 independent latent variable 

    # load test images
    print("loading images and embedding 0.")
    img_path = os.path.join(args.images_dir, "0.png")
    image0 = image_reader(img_path)
    image0 = image0.to(device)
    w_latent0, syn_img = embedding_to_Wplus(image0, "0")

    print("loading images and embedding 4.")
    img_path = os.path.join(args.images_dir, "4.png")
    image4 = image_reader(img_path)
    image4 = image4.to(device)
    w_latent4, syn_img = embedding_to_Wplus(image4, "4")
    plt.subplot(1,2,1)
    plt.imshow(image4.detach().cpu().squeeze().permute((1,2,0)).numpy())
    plt.title("input")
    plt.subplot(1,2,2)
    plt.imshow(syn_img.detach().cpu().squeeze().permute((1,2,0)).numpy())
    plt.title("reconst.")
    plt.show()

    print("loading images and embedding 5.")
    img_path = os.path.join(args.images_dir, "5.png")
    image5 = image_reader(img_path)
    image5 = image5.to(device)
    w_latent5, syn_img = embedding_to_Wplus(image5, "5")
    plt.subplot(1,2,1)
    plt.imshow(image5.detach().cpu().squeeze().permute((1,2,0)).numpy())
    plt.title("input")
    plt.subplot(1,2,2)
    plt.imshow(syn_img.detach().cpu().squeeze().permute((1,2,0)).numpy())
    plt.title("reconst.")
    plt.show()

    print("loading images and embedding 7.")
    img_path = os.path.join(args.images_dir, "7.png")
    image7 = image_reader(img_path)
    image7 = image7.to(device)
    w_latent7, syn_img = embedding_to_Wplus(image7, "7")
    print(f"size of w latent: {w_latent7.size()}")

    # morphing test 
    print("morping starts")
    nsteps = 10
    for i in range(nsteps+1):
        alpha = i/nsteps
        _, syn_img = morphing_in_Wplus(w_latent4, w_latent5, i/nsteps)
        file_path = "save_images/image2stylegan/morphing/morphed_{}_{}_{}.png".format(4,5,int(alpha*10))
        save_image(syn_img.clamp(0, 1), file_path) 
    alpha = 0.5
    w_latent_morphed0, syn_img  = morphing_in_Wplus(w_latent4, w_latent5, 0.5)
    morphed_path = "save_images/image2stylegan/morphing/morphed_{}_{}_{}.png".format(4,5,int(alpha*10))
    save_image(syn_img.clamp(0, 1), morphed_path) 
    print("original morphed w:", w_latent_morphed0)
    print("morping done")
    
    ##############################################################
    # demopring =  morphing - 5
    ##############################################################
    print("demorping starts")
    print("loading and embedding the moprphed image")
    image_morphed = image_reader(morphed_path)
    image_morphed = image_morphed.to(device)
    w_latent_morphed1,_ = embedding_to_Wplus(image_morphed)
    print("recon morphed w:", w_latent_morphed1)

    w_diff = w_latent_morphed1 - w_latent_morphed0
    print("diff morphed w:", w_diff)

    print("loading and embedding the live image")
    img_path = os.path.join(args.images_dir, "5.png")
    image_live = image_reader(img_path)
    image_live = image_live.to(device)
    w_latent_live, _ = embedding_to_Wplus(image_live)

    print("demorphing")
    syn_img = demorph_in_Wplus(w_latent_morphed0, w_latent5) 
    plt.subplot(1,4,1)
    plt.imshow(syn_img.detach().cpu().squeeze().permute((1,2,0)).numpy())
    file_path = "save_images/image2stylegan/morphing/{}.png".format("demorph0")
    save_image(syn_img.clamp(0, 1), file_path)
    syn_img = demorph_in_Wplus(w_latent_morphed1, w_latent_live) 
    plt.subplot(1,4,2)
    plt.imshow(syn_img.detach().cpu().squeeze().permute((1,2,0)).numpy())
    file_path = "save_images/image2stylegan/morphing/{}.png".format("demorph1")
    save_image(syn_img.clamp(0, 1), file_path)
    syn_img = demorph_in_Wplus(w_latent_morphed0, w_latent_live) 
    plt.subplot(1,4,3)
    plt.imshow(syn_img.detach().cpu().squeeze().permute((1,2,0)).numpy())
    file_path = "save_images/image2stylegan/morphing/{}.png".format("demorph2")
    save_image(syn_img.clamp(0, 1), file_path)
    syn_img = demorph_in_Wplus(w_latent_morphed1, w_latent5) 
    plt.subplot(1,4,4)
    plt.imshow(syn_img.detach().cpu().squeeze().permute((1,2,0)).numpy())
    file_path = "save_images/image2stylegan/morphing/{}.png".format("demorph3")
    save_image(syn_img.clamp(0, 1), file_path)
    plt.show()

    ##############################################################
    # demopring =  morphing - 4
    ##############################################################


    ##############################################################
    # demopring =  5 - 5
    ##############################################################

    ##############################################################
    # demopring =  4 - 4
    ##############################################################

    print("demorphing done")


    # style transfer 
    print("style transfer starts")
    style_transfer(w_latent0, w_latent7, 0, 7)
    print("style transfer done")
   
    print("Success in all")


if __name__ == "__main__":

    # 0. setting parameter
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cuda", action='store', default=True, type=bool)
    parser.add_argument("--use_noise", action='store', default=True, type=bool)  # turn on/off the StyleGan detail noise input 
    parser.add_argument("--model_dir", action='store', default="pretrain_stylegan", type=str)
    parser.add_argument("--model_name", action='store', default="karras2019stylegan-ffhq-1024x1024.pt", type=str)
    parser.add_argument("--images_dir", action='store', default="images/image2stylegan", type=str)
    parser.add_argument("--lr", action='store', default=0.01, type=float)
    parser.add_argument("--epochs", action='store', default=2000, type=int)   # iteration in fact 
    args = parser.parse_args()


    #1. loading StyleGan Model, Synthesis network and Mapping network
    device = get_device(args.use_cuda)
    g_all = nn.Sequential(OrderedDict([('g_mapping', G_mapping()), ('g_synthesis', G_synthesis(use_noise = args.use_noise, resolution=1024)) ]))
    # Load the pre-trained model
    g_all.load_state_dict(torch.load(os.path.join(args.model_dir, args.model_name), map_location=device), strict = not args.use_noise)
    g_all.eval()
    g_all.to(device)
    g_mapping, g_synthesis = g_all[0], g_all[1]
    print("Success in loading models")

    ############################################
    # Test 
    #############################################
    
    # make average W vector 
    mean_W = make_mean_latent_W()
    print(f"meanW:{mean_W}")

    # check if we can get the same vector for the same image 
    check_repeatibility()

    # main test for morphing and demorping 
    image2StyleGan_W()

    #image2StyleGan_Wplus()


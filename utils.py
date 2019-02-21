import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
import PIL
import torch.nn as nn
from config import opt


class TextureDataset(Dataset):
    """Dataset wrapping images from a random folder with textures

    Arguments:
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, img_path, transform=None,scale=1):
        self.img_path = img_path
        self.transform = transform    
        if True:##ok this is for 1 worker only!
            names = os.listdir(img_path)
            self.X_train =[]
            for n in names:
                name =self.img_path + n
                try:
                    img = Image.open(name)
                    try:
                        img = img.convert('RGB')##fixes truncation???
                    except:
                        pass
                    if scale!=1:
                        img=img.resize((int(img.size[0]*scale),int(img.size[1]*scale)),PIL.Image.LANCZOS)
                except Exception as e:
                    print (e,name)
                    continue

                self.X_train +=[img]
                print (n,"img added", img.size,"total length",len(self.X_train))
                if len(self.X_train) > 4000:
                    break ##usually want to avoid so many files

        ##this affects epoch length..
        if len(self.X_train) < 2000:
            c = int(2000/len(self.X_train))
            self.X_train*=c

    def __getitem__(self, index):
        if False:
            name =self.img_path + self.X_train[index]
            img = Image.open(name)
        else:
            img= self.X_train[index]#np.random.randint(len(self.X_train))   
        if self.transform is not None:
            img2 = self.transform(img)        
        label =0
        #print ('data returned',img2.data.shape)
        return img2, label

    def __len__(self):
        return len(self.X_train)

def GaussKernel(sigma,wid=None):
    if wid is None:
        wid =2 * 2 * sigma + 1+10

    def gaussian(x, mu, sigma):
        return np.exp(-(float(x) - float(mu)) ** 2 / (2 * sigma ** 2))
    def make_kernel(sigma):
        # kernel radius = 2*sigma, but minimum 3x3 matrix
        kernel_size = max(3, int(wid))
        kernel_size = min(kernel_size,150)
        mean = np.floor(0.5 * kernel_size)
        kernel_1d = np.array([gaussian(x, mean, sigma) for x in range(kernel_size)])
        # make 2D kernel
        np_kernel = np.outer(kernel_1d, kernel_1d).astype(dtype=np.float32)
        # normalize kernel by sum of elements
        kernel = np_kernel / np.sum(np_kernel)
        return kernel
    ker = make_kernel(sigma)
  
    a = np.zeros((3,3,ker.shape[0],ker.shape[0])).astype(dtype=np.float32)
    for i in range(3):
        a[i,i] = ker
    return a

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gsigma=1.##how much to blur - larger blurs more ##+"_sig"+str(gsigma)
gwid=61
kernel = torch.FloatTensor(GaussKernel(gsigma,wid=gwid)).to(device)##slow, pooling better
def avgP(x):
    return nn.functional.avg_pool2d(x,int(16))
def avgG(x):
    pad=nn.functional.pad(x,(gwid//2,gwid//2,gwid//2,gwid//2),'reflect')##last 2 dimensions padded
    return nn.functional.conv2d(pad,kernel)##reflect pad should avoid border artifacts

def plotStats(a,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15,15))
    names = ["pTrue", "pFake", "pFake2", "contentLoss I", "contentLoss I_M", "norm(alpha)", "entropy(A)", "tv(A)", "tv(alpha)", "diversity(A)"]
    win=50##for running avg
    for i in range(a.shape[1]):
        if i <3:
            ix=0
        elif i <5:
            ix =1
        elif i >=5:
            ix=i-3
        plt.subplot(a.shape[1]-3+1,1,ix+1)
        plt.plot(a[:,i],label= "err"+str(i)+"_"+names[i])
        try:
            av=np.convolve(a[:,i], np.ones((win,))/win, mode='valid')
            plt.plot(av,label= "av"+str(i)+"_"+names[i],lw=3)
        except Exception as e:
            print ("ploterr",e)
        plt.legend(loc="lower left")
    plt.savefig(path+"plot.png")

    def Mstring(v):
        s=""
        for i in range(v.shape[0]):
            s+= names[i]+" "+str(v[i])+";"
        return s

    print("MEAN",Mstring(a.mean(0)))
    print("MEAN",Mstring(a[-100:].mean(0)))
    plt.close()

#large alpha emphasizes new -- conv. generation , less effect on old, the mix template output
#@param I_G is parametric generation
#@param I_M is mixed template image
def blend(I_G, I_M, alpha, beta):
    if opt.blendMode==0:
        out= I_M*(1 - beta) + alpha * I_G[:, :3]
    if opt.blendMode==1:
        out = I_G[:, :3] * alpha * 2 + I_M
    if opt.blendMode==2:##this is the mode described in paper, convex combination
        out= I_G[:, :3] * alpha + (1 - alpha) * I_M
    return torch.clamp(out,-1,1)

##show the different btw final image and mixed image -- this shows the parametric output of our network
def invblend(I,I_M,alpha,beta):
    return torch.clamp(I-I_M,-1,1)

#absolute difference in X and Y directions
def total_variation(y):
    return torch.mean(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:])) + torch.mean(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :]))

##2D array of the edges of C channels image
def tvArray(x):
    border1 = x[:, :, :-1] - x[:, :, 1:]
    border1 = torch.cat([border1.abs().sum(1).unsqueeze(1), x[:, :1, :1] * 0], 2)  ##so square with extra 0 line
    border2 = x[:, :, :, :-1] - x[:, :, :, 1:]
    border2 = torch.cat([border2.abs().sum(1).unsqueeze(1), x[:, :1, :, :1] * 0], 3)
    border = torch.cat([border1, border2], 1)
    return border

##negative gram matrix
def gramMatrix(x,y=None,sq=True,bEnergy=False):
    if y is None:
        y = x

    B, CE, width, height = x.size()
    hw = width * height

    energy = torch.bmm(x.permute(2, 3, 0, 1).view(hw, B, CE),
                       y.permute(2, 3, 1, 0).view(hw, CE, B), )
    energy = energy.permute(1, 2, 0).view(B, B, width, height)
    if bEnergy:
        return energy
    sqX = (x ** 2).sum(1).unsqueeze(0)
    sqY = (y ** 2).sum(1).unsqueeze(1)
    d=-2 * energy + sqX + sqY
    if not sq:
        return d##debugging
    gram = -torch.clamp(d, min=1e-10)#.sqrt()
    return gram

##some image level content loss
def contentLoss(a,b,netR,opt):
    def nr(x):
        return (x**2).mean()
        return x.abs().mean()

    if opt.cLoss==0:
        a = avgG(a)
        b = avgG(b)
        return nr(a.mean(1) - b.mean(1))
    if opt.cLoss==1:
        a = avgP(a)
        b = avgP(b)
        return nr(a.mean(1) - b.mean(1))

    if opt.cLoss==10:
        return nr(netR(a)-netR(b))

    if opt.cLoss==100:
        return nr(netR(a)-b)
    if opt.cLoss == 101:
        return nr(avgG(netR(a)) - avgG(b))
    if opt.cLoss == 102:
        return nr(avgP(netR(a)) - avgP(b))
    if opt.cLoss == 103:
        return nr(avgG(netR(a)).mean(1) - avgG(b).mean(1))

    raise Exception("NYI")

##visualization routine to show mix arrayA as many colourful channels
def rgb_channels(x):
    N=x.shape[1]
    if N ==1:
        return torch.cat([x,x,x],1)##just white dummy

    cu= int(N**(1/3.0))+1
    a=x[:,:3]*0##RGB image
    for i in range(N):
        c1=int(i%cu)
        j=i//cu
        c2=int(j%cu)
        j=j//cu
        c3=int(j%cu)
        a[:,:1]+= c1/float(cu+1)*x[:,i].unsqueeze(1)
        a[:,1:2]+=c2/float(cu+1)*x[:,i].unsqueeze(1)
        a[:,2:3]+=c3/float(cu+1)*x[:,i].unsqueeze(1)
    return a#*2-1##so 0 1


if opt.zPeriodic:
    # 2*nPeriodic initial spread values
    # slowest wave 0.5 pi-- full cycle after 4 steps in noise tensor
    # fastest wave 1.5pi step -- full cycle in 0.66 steps
    def initWave(nPeriodic):
        buf = []
        for i in range(nPeriodic // 4+1):
            v = 0.5 + i / float(nPeriodic//4+1e-10)
            buf += [0, v, v, 0]
            buf += [0, -v, v, 0]  # #so from other quadrants as well..
        buf=buf[:2*nPeriodic]
        awave = np.array(buf, dtype=np.float32) * np.pi
        awave = torch.FloatTensor(awave).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        return awave
    waveNumbers = initWave(opt.zPeriodic).to(device)

    class Waver(nn.Module):
        def __init__(self):
            super(Waver, self).__init__()
            if opt.zGL >0:
                K=50
                layers=[nn.Conv2d(opt.zGL, K, 1)]
                layers +=[nn.ReLU(True)]
                layers += [nn.Conv2d(K,2*opt.zPeriodic, 1)]
                self.learnedWN =  nn.Sequential(*layers)
            else:##static
                self.learnedWN = nn.Parameter(torch.zeros(opt.zPeriodic * 2).uniform_(-1, 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0) * 0.2)
        def forward(self, c,GLZ=None):
            if opt.zGL > 0:
                return (waveNumbers + 5*self.learnedWN(GLZ)) * c

            return (waveNumbers + self.learnedWN) * c
    learnedWN = Waver()
else:
    learnedWN = None

##inplace set noise
def setNoise(noise):
    noise=noise.detach()*1.0
    noise.uniform_(-1, 1)  # normal_(0, 1)
    if opt.zGL:
        noise[:, :opt.zGL] = noise[:, :opt.zGL, :1, :1].repeat(1, 1,noise.shape[2],noise.shape[3])
    if opt.zPeriodic:
        xv, yv = np.meshgrid(np.arange(noise.shape[2]), np.arange(noise.shape[3]),indexing='ij')
        c = torch.FloatTensor(np.concatenate([xv[np.newaxis], yv[np.newaxis]], 0)[np.newaxis])
        c = c.repeat(noise.shape[0], opt.zPeriodic, 1, 1)
        c = c.to(device)
        # #now c has canonic coordinate system -- multiply by wave numbers
        raw = learnedWN(c,noise[:, :opt.zGL])
        #random offset
        offset = (noise[:, -opt.zPeriodic:, :1, :1] * 1.0).uniform_(-1, 1) * 6.28
        offset = offset.repeat(1, 1, noise.shape[2], noise.shape[3])
        wave = torch.sin(raw[:, ::2] + raw[:, 1::2] + offset)
        noise[:,-opt.zPeriodic:]=wave
    return noise

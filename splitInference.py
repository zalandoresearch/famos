from config import nDep as gen_ls
import torch
import torch.nn as nn
import sys
import gc

nUp_n = 2 ** gen_ls ##noise and image tensor relation
nUp   = 2**(gen_ls+1) ##for buffer ovelap calculation, >=nUp_n
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def splitH(Im_tri, noise,te,f,sH):
    S = Im_tri.shape[3]
    orig = 0 * Im_tri[:, :3]
    if te is not None:
        alpha = 0 * Im_tri[:, :3]
        origM=orig*0
        blenda = 0 * Im_tri[:, :1].repeat(1, te.shape[1], 1, 1)##te.shape[1] is the template count N
    else:
        alpha=None
        origM=None
        blenda=None

    err = 0
    increment = []
    oldS = 0
    s=sH
    for i in range(1, s + 1):
        S1 = int(i / float(s) * S)
        S1 = S1 - S1 % nUp
        increment += [(oldS, S1)]
        oldS = S1

    print ("H increment", increment,"image",Im_tri.shape,"noise",noise.shape)

    def _proc(incr):

        if incr[0] == 0:
            li = 0
            lz = 0
        else:
            li = incr[0] - 4 * nUp
            lz = li//nUp_n#incr[0] // nUp - 4
        if incr[1] == S:
            ui = S
            uz = S // nUp_n
        else:
            ui = incr[1] + 4 * nUp
            uz = ui//nUp_n#incr[1] // nUp + 4

        #print (li, ui, "z", lz, uz,"incr", incr,)

        Im_tri1 = Im_tri[:, :, :, li:ui]
        noise1 = noise[:, :, :, lz:uz]
        Im_tri1 = Im_tri1.to(device)

        #print ("H values",Im_tri1.shape,noise1.shape,"idxes",li,ui,lz,uz)

        if te is not None:
            te1 = te[:, :, :, :, li:ui].float()
            te1=te1.to(device)##more engineered but efficient in code: only add template chunk to memory, not full large template batch
        else:
            te1=None

        gen1,alpha1,blenda1,mix1 = f(Im_tri1, noise1,te1,True)

        #print ("setting indices", gen1[:, :, :, incr[0] - li:incr[1] - li].shape, orig[:, :, :, incr[0]:incr[1]].shape)
        orig[:, :, :, incr[0]:incr[1]] = gen1[:, :, :, incr[0] - li:incr[1] - li]
        if te is not None:
            alpha[:, :, :, incr[0]:incr[1]] = alpha1[:, :, :, incr[0] - li:incr[1] - li]
            origM[:, :, :, incr[0]:incr[1]] = mix1[:, :, :, incr[0] - li:incr[1] - li]
            blenda[:, :, :, incr[0]:incr[1]] =blenda1[:, :, :, incr[0] - li:incr[1] - li]
        gc.collect()##hmm, doe not help
        torch.cuda.empty_cache()
        return 0#error_full_1

    for incr in increment:
        err += _proc(incr)
        sys.stdout.flush()

    return orig,alpha,blenda,origM, err  # error_full_1+error_full_2


def splitW(Im_tri, noise,te,f):
    ##careful with size: too small and will run out of memory; too large and will have empty slice and cause bug
    ##some rough heuristic how to choose size
    sW = Im_tri.shape[2]//480
    sH = Im_tri.shape[3]//480
    print ("generated split ratios",sW,sH)

    S = Im_tri.shape[2]
    ##4 image buffers
    orig = 0 * Im_tri[:, :3]
    if te is not None:
        alpha=0 * Im_tri[:, :3]
        origM=orig*0
        blenda = 0*Im_tri[:, :1].repeat(1, te.shape[1], 1, 1)##te.shape[1] is the template count
    else:
        alpha = None
        origM = None
        blenda = None

    err = 0
    increment = []
    s = sW
    oldS = 0
    for i in range(1, s + 1):
        S1 = int(i / float(s) * S)
        S1 = S1 - S1 % nUp
        increment += [(oldS, S1)]
        oldS = S1
    print ("W increment", increment,"image",Im_tri.shape,"noise",noise.shape)
    def _proc(incr):

        if incr[0] == 0:
            li = 0
            lz = 0
        else:
            li = incr[0] - 4 * nUp
            lz = li//nUp_n#incr[0] // nUp - 4
        if incr[1] == S:
            ui = S
            uz = S // nUp_n
        else:
            ui = incr[1] + 4 * nUp
            uz = ui//nUp_n#incr[1] // nUp + 4

        #print (li, ui, "z", lz, uz)
        #print ("incr", incr,)

        Im_tri1 = Im_tri[:, :, li:ui]
        noise1 = noise[:, :, lz:uz]

        if te is not None:
            te1 = te[:, :, :, li:ui]
        else:
            te1=None
        gen1,alpha1,blenda1,mix1, error_full_1 = splitH(Im_tri1, noise1,te1,f,sH)

        #print ("setting indices", gen1[:, :, incr[0] - li:incr[1] - li].shape, orig[:, :, incr[0]:incr[1]].shape)
        orig[:, :, incr[0]:incr[1]] = gen1[:, :, incr[0] - li:incr[1] - li]
        if te is not None:
            alpha[:, :, incr[0]:incr[1]] = alpha1[:, :,  incr[0] - li:incr[1] - li]
            blenda[:, :,incr[0]:incr[1]] = blenda1[:, :, incr[0] - li:incr[1] - li]
            origM[:, :, incr[0]:incr[1]] = mix1[:, :, incr[0] - li:incr[1] - li]
        return error_full_1
    for incr in increment:
        err += _proc(incr)
    return orig,alpha,blenda,origM

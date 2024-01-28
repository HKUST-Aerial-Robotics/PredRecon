from models import PCN, SPM, ScaleEst, SPM_distill
import torch
import argparse

def sph(params):
    torch.set_grad_enabled(False)

    spm_net = SPM_distill().cuda()
    spm_net.eval()
    spm_net.load_state_dict(torch.load(params.ckpt_path))

    example = torch.rand(1,8192,3).cuda()

    torch.jit.trace(spm_net, example).save(params.save_path)

def seh(params):

    torch.set_grad_enabled(False)

    seh_net = ScaleEst().cuda()
    seh_net.eval()
    seh_net.load_state_dict(torch.load(params.ckpt_path))

    example = torch.rand(1,8192,3).cuda()

    torch.jit.trace(seh_net, example).save(params.save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Convert')
    parser.add_argument('--ckpt_path', type=str, help='checkpoint of tracing model')
    parser.add_argument('--header', type=str, help='which header of SPM')
    parser.add_argument('--save_path', type=str, help='save path of traced model')
    params = parser.parse_args()

    if params.header == 'seh':
        seh(params)
    if params.header == 'sph':
        sph(params)
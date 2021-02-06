"""
Code adapted from https://github.com/richzhang/PerceptualSimilarity

Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import perceptual_models
import numpy
import os
from utils import im2tensor, load_image
import torch

use_gpu = True
use_gpu = use_gpu and torch.cuda.is_available()

def LPIPS_cal(dataset_teacher, number_classes, image_dir, output_file, device):
    print("Calculate LPIPS value...")
    all_pairs_lpips = False
    model = perceptual_models.PerceptualLoss(model='net-lin', net='alex', use_gpu=use_gpu, version=0.1)
    dists_among_fake_imgs = []
    for i in range(number_classes):
        files = os.listdir(image_dir + dataset_teacher + '/fake_images/' + str(i))
        num = len(files)-1
        if num > 10:
            num = 10
        for (ff, file0) in enumerate(files[0:num]):  # [:-1]):
            img0 = im2tensor(
                load_image(os.path.join(image_dir + dataset_teacher + '/fake_images/' + str(i), file0))).to(device)  # RGB image from [-1,1]
            if (all_pairs_lpips):
                files1 = files[ff + 1:]
            else:
                files1 = [files[ff + 1], ]
            for file1 in files1[0:num]:  # :
                img1 = im2tensor(
                    load_image(os.path.join(image_dir + dataset_teacher + '/fake_images/' + str(i), file1))).to(
                    device)  # RGB image from [-1,1]
                # Compute distance
                dist01 = model.forward(img0, img1)
                dists_among_fake_imgs.append(dist01.item())
    avg_dist_among_fake_imgs = numpy.mean(numpy.array(dists_among_fake_imgs))
    stderr_dist_among_fake_imgs = numpy.std(numpy.array(dists_among_fake_imgs))/numpy.sqrt(len(dists_among_fake_imgs))
    print("[LPIPS_mean: %s] [LPIPS_std: %s] " % (
        avg_dist_among_fake_imgs, stderr_dist_among_fake_imgs))
    with open(output_file, 'a') as f:
        f.write(str(avg_dist_among_fake_imgs) + '\n')
        f.close()

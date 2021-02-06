from generator import *
from utils import *
import argparse
from torch.autograd import Variable
import os
set_seed(seed)

parser = argparse.ArgumentParser(description='train-generator-network')
parser.add_argument('--teacher_dir', type=str, default='./Model/')
parser.add_argument('--lr_G', type=float, default=0.001)
parser.add_argument('--data_dir', type=str, default='./Data/')
parser.add_argument('--dataset_teacher', type=str, default='MNIST')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--latent_dim', type=int, default=100)
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--output_dir', type=str, default='./Model/')
parser.add_argument('--teacher_name', type=str, default='teacher')
parser.add_argument('--generator_name', type=str, default='generator')
parser.add_argument('--output_file', type=str, default=os.path.join(os.path.dirname(__file__), 'results/') + 'results_generator.csv')

args, unknown = parser.parse_known_args()

generator = Generator(img_size=args.img_size, latent_dim=args.latent_dim, channels=args.channels).to(device)  # Generator
criterion = torch.nn.CrossEntropyLoss().to(device)
generator = nn.DataParallel(generator).to(device)
if use_gpu:
    teacher = torch.load(args.teacher_dir + args.teacher_name).to(device)  # Teacher
else:
    teacher = torch.load(args.teacher_dir + args.teacher_name, map_location=torch.device('cpu')).to(device)  # Teacher
teacher.eval()
teacher = nn.DataParallel(teacher).to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr_G)
data_test_loader, data_test = load_data_test_loader(args.data_dir, args.dataset_teacher, args.batch_size)  # Teacher Test Data

# ---------------------------------------------------------------------------------------------------
#  Training
# ----------------------------------------------------------------------------------------------------
with open(args.output_file, 'a') as f:
    f.write(
        'Epoch, Iter, LossOH, LossIE, LossDS, LossGenerator \n')
    f.close()
loss_oh_prev = 0.0
loss_ie_pre = 0.0
for epoch in range(args.n_epochs):
    for i in range(120):
        generator.train()  # have effect
        z1 = Variable(torch.randn(int(args.batch_size/2), args.latent_dim)).to(device)
        z2 = Variable(torch.randn(int(args.batch_size/2), args.latent_dim)).to(device)
        z = torch.cat((z1, z2), 0)
        optimizer_G.zero_grad()
        gen_imgs, state_G = generator(z)
        gen_imgs1, gen_imgs2 = torch.split(gen_imgs, z1.size(0), dim=0)
        outputs_T1, features_T1, *activations_T1 = teacher(gen_imgs1, out_feature=True, out_activation=True)
        outputs_T2, features_T2, *activations_T2 = teacher(gen_imgs2, out_feature=True, out_activation=True)

        outputs_T = torch.cat((outputs_T1, outputs_T2))

        #  one-hot loss
        pred = outputs_T.data.max(1)[1]
        loss_one_hot = criterion(outputs_T, pred)

        # information entropy loss
        mean_softmax_T = torch.nn.functional.softmax(outputs_T, dim=1).mean(dim=0)
        loss_information_entropy = (mean_softmax_T * torch.log(mean_softmax_T)).sum()

        softmax_o_T1 = torch.nn.functional.softmax(outputs_T1, dim=1)
        softmax_o_T2 = torch.nn.functional.softmax(outputs_T2, dim=1)
        lz = torch.norm(gen_imgs2 - gen_imgs1) / torch.norm(softmax_o_T2 - softmax_o_T1)
        loss_diversity_seeking = 1 / (lz + 1 * 1e-20)
        loss = torch.exp(loss_one_hot - loss_oh_prev) + torch.exp(loss_information_entropy - loss_ie_pre) + loss_diversity_seeking

        loss.backward()
        optimizer_G.step()

        if i == 1:
            loss_oh_prev = loss_one_hot.detach()
            loss_ie_pre = loss_information_entropy.detach()
            print("[Epoch %d/%d] [loss_kd: %f] " % (epoch, args.n_epochs, loss.item()))
        with open(args.output_file, 'a') as f:
            f.write(str(epoch) + ',' + str(i) + ',' + str(loss_one_hot.item()) + ',' + str(loss_information_entropy.item()) + ','
                    + str(loss_diversity_seeking.item()) + ',' + str(loss.item()) + '\n')
            f.close()

torch.save(generator, args.output_dir + args.generator_name)

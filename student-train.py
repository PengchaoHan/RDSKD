from utils import *
from torch.autograd import Variable
import argparse
import os

set_seed(seed)

parser = argparse.ArgumentParser(description='train-student-network')
parser.add_argument('--teacher_dir', type=str, default='./Model/')
parser.add_argument('--generator_dir', type=str, default='./Model/')
parser.add_argument('--student_model', type=str, default='LeNet5Half')
parser.add_argument('--lr_S', type=float, default=0.002)
parser.add_argument('--data_dir', type=str, default='./Data/')
parser.add_argument('--dataset_teacher', type=str, default='MNIST')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--latent_dim', type=int, default=100)
parser.add_argument('--output_dir', type=str, default='./Model/')
parser.add_argument('--kl_temperature', type=int, default=10)
parser.add_argument('--teacher_name', type=str, default='teacher')
parser.add_argument('--generator_name', type=str, default='generator')
parser.add_argument('--student_name', type=str, default='student')
parser.add_argument('--output_file', type=str, default=os.path.join(os.path.dirname(__file__), 'results/') + 'results_student.csv')


args, unknown = parser.parse_known_args()

criterion = torch.nn.CrossEntropyLoss().to(device)
if use_gpu:
    teacher = torch.load(args.teacher_dir + args.teacher_name).to(device)  # Teacher
    generator = torch.load(args.generator_dir + args.generator_name).to(device)  # Generator
else:
    teacher = torch.load(args.teacher_dir + args.teacher_name, map_location=torch.device('cpu')).to(device)  # Teacher
    generator = torch.load(args.generator_dir + args.generator_name, map_location=torch.device('cpu')).to(device)  # Generator
teacher.eval()
teacher = nn.DataParallel(teacher).to(device)
generator.eval()
generator = nn.DataParallel(generator).to(device)

student_net, optimizer_S = load_student(args.student_model, args.lr_S, teacher_dataset=args.dataset_teacher)  # Student
data_test_loader, data_test = load_data_test_loader(args.data_dir, args.dataset_teacher, args.batch_size)  # Teacher Test Data


# ---------------------------------------------------------------------------------------------------
#  Training
# ----------------------------------------------------------------------------------------------------
with open(args.output_file, 'a') as f:
    f.write(
        'Epoch, Iter, TrainLossKD, TestLossKD, TestAccuracy\n')
    f.close()
for epoch in range(args.n_epochs):
    if args.dataset_teacher == 'cifar10':
        adjust_learning_rate(optimizer_S, epoch, args.lr_S)

    for i in range(120):
        student_net.train()
        z = Variable(torch.randn(args.batch_size, args.latent_dim)).to(device)
        optimizer_S.zero_grad()
        gen_imgs, state_G = generator(z)
        outputs_T, features_T = teacher(gen_imgs, out_feature=True)
        loss_kd = kdloss(student_net(gen_imgs.detach()) / args.kl_temperature, outputs_T.detach() / args.kl_temperature)
        loss = args.kl_temperature * args.kl_temperature * loss_kd
        loss.backward()
        optimizer_S.step()

        if i % 10 == 0:
            test_loss, test_accr, pred_real = model_test(student_net, data_test, data_test_loader, criterion)
            print("[Epoch %d/%d] [loss_kd: %f] [Test accuracy: %f]" % (epoch, args.n_epochs, loss_kd.item(), test_accr))
            with open(args.output_file, 'a') as f:
                f.write(str(epoch) + ',' + str(i) + ',' + str(loss_kd.item()) + ',' + str(test_loss) + ',' + str(test_accr) + '\n')
                f.close()

    torch.save(student_net, args.output_dir + args.student_name)

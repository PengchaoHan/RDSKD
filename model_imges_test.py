from inception_score import inception_score
from fid_score import calculate_fid_given_images
from lpips_score import LPIPS_cal
from utils import *
import imageio
import shutil
import os
from os import getcwd
from pandas import read_csv
from os import path
import argparse
import numpy as np
from torch.autograd import Variable

set_seed(seed)

global data_test_loader
global gen_imgs
global pred
global student
global generator
global teacher
global criterion
global data_test

parser = argparse.ArgumentParser(description='train-student-network')
parser.add_argument('--teacher_dir', type=str, default='./Model/')
parser.add_argument('--student_dir', type=str, default='./Model/')
parser.add_argument('--generator_dir', type=str, default='./Model/')
parser.add_argument('--data_dir', type=str, default='./Data/')
parser.add_argument('--image_dir', type=str, default='./images/')
parser.add_argument('--dataset_teacher', type=str, default='MNIST')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--channels', type=int, default=1)
parser.add_argument('--latent_dim', type=int, default=100)
parser.add_argument('--number_classes', type=int, default=10)
parser.add_argument('--img_size', type=int, default=32)
parser.add_argument('--teacher_name', type=str, default='teacher')
parser.add_argument('--generator_name', type=str, default='generator')
parser.add_argument('--student_name', type=str, default='student')
parser.add_argument('--output_file', type=str, default=os.path.join(os.path.dirname(__file__), 'results/') + 'results.csv')

args, unknown = parser.parse_known_args()

def get_teacher_model():
    global teacher
    if use_gpu:
        teacher = torch.load(args.teacher_dir + args.teacher_name).to(device)  # Teacher
    else:
        teacher = torch.load(args.teacher_dir + args.teacher_name, map_location=torch.device('cpu')).to(device)  # Teacher
    teacher.eval()
    teacher = nn.DataParallel(teacher).to(device)

def get_student_model():
    global student
    if use_gpu:
        student = torch.load(args.student_dir + args.student_name).to(device)  # Student
    else:
        student = torch.load(args.student_dir + args.student_name, map_location=torch.device('cpu')).to(device)  # Student
    student.eval()
    student = nn.DataParallel(student).to(device)

def get_dataset():
    global data_test_loader
    global data_test
    global criterion
    criterion = torch.nn.CrossEntropyLoss().to(device)
    data_test_loader, data_test = load_data_test_loader(args.data_dir, args.dataset_teacher, args.batch_size)

def gen_images():
    global generator
    global gen_imgs
    global pred
    global features_T
    global activations_T
    get_teacher_model()
    if use_gpu:
        generator = torch.load(args.generator_dir + args.generator_name).to(device)  # Generator
    else:
        generator = torch.load(args.generator_dir + args.generator_name, map_location=torch.device('cpu')).to(device)  # Generator
    generator.eval()
    generator = nn.DataParallel(generator).to(device)
    with torch.no_grad():
        z = Variable(torch.randn(args.batch_size, args.latent_dim)).to(device)
        gen_imgs, state_G = generator(z)
        outputs_T = teacher(gen_imgs)
        pred = outputs_T.data.max(1)[1]

def save_real_images():
    print("Save true images....")
    for i in range(args. number_classes):  # delete existing images
        if os.path.exists(args.image_dir + args.dataset_teacher + '/real_images/' + str(i)):
            shutil.rmtree(args.image_dir + args.dataset_teacher + '/real_images/' + str(i))
        os.mkdir(args.image_dir + args.dataset_teacher + '/real_images/' + str(i))
    true_images = []
    for i, (images, labels) in enumerate(data_test_loader):
        for j in range(len(images)):  # save images to files for LPIPS calculation
            if args.channels == 1:
                img = images[j].detach()[0].to(torch.device('cpu'))
                imageio.imwrite(args.image_dir + args.dataset_teacher + '/real_images/' + str(labels[j].item()) + '/img_' + str(j) + '.jpg', img)
            if args.channels == 3:
                img = images[j].detach().to(torch.device('cpu'))
                img = np.array(img).transpose((1, 2, 0))  # No dtype=np.uint8 !!
                imageio.imwrite(args.image_dir + args.dataset_teacher + '/real_images/' + str(labels[j].item()) + '/img_' + str(j) + '.jpg', img)

        true_images.extend(images)
        if len(true_images) > args.batch_size:  # NOT enough memory
            break

def save_fake_images():
    with torch.no_grad():
        print("Save fake images....")
        for i in range(args.number_classes):  # delete existing images
            if os.path.exists(args.image_dir + args.dataset_teacher + '/fake_images/' + str(i)):
                shutil.rmtree(args.image_dir + args.dataset_teacher + '/fake_images/' + str(i))
            os.mkdir(args.image_dir + args.dataset_teacher + '/fake_images/' + str(i))
        for j in range(len(gen_imgs)):  # save images to files for LPIPS calculation
            if args.channels == 1:
                img = gen_imgs[j].detach()[0].to(torch.device('cpu'))
                imageio.imwrite(args.image_dir + args.dataset_teacher + '/fake_images/' + str(pred[j].item()) + '/img_' + str(j) + '.jpg', img)
            if args.channels == 3:
                img = gen_imgs[j].detach().to(torch.device('cpu'))
                img = np.array(img).transpose((1, 2, 0))  # No dtype=np.uint8 !!
                imageio.imwrite(args.image_dir + args.dataset_teacher + '/fake_images/' + str(pred[j].item()) + '/img_' + str(j) + '.jpg', img)

def test_teacher():
    print("Teacher test...")
    test_loss_t, test_accr_t, _ = model_test(teacher, data_test, data_test_loader, criterion)

    with open(args.output_file, 'a') as f:
        f.write(args.dataset_teacher + ',' + str(test_accr_t) + ',')
        f.close()

def test_student():
    print("Student test...")
    test_loss, test_accr, _ = model_test(student, data_test, data_test_loader, criterion)

    with open(args.output_file, 'a') as f:
        f.write(str(test_accr) + ',')
        f.close()

def test_results():
    length = 100
    result_csv_colnames = ['Epoch', 'Iter', 'TrainLossKD', 'TestLossKD', 'TestAccuracy']
    f = 'results_student.csv'
    fp = path.join(getcwd()+'/results', f)
    results_df = read_csv(fp, header=None, index_col=False, names=result_csv_colnames)

    kd_loss = results_df['TrainLossKD'].tolist()[1:]
    kd_loss = np.array([float(kd_loss[i]) for i in range(len(kd_loss)) if i > len(kd_loss) - length])
    kd_loss_mean = np.mean(kd_loss)
    kd_loss_std = np.std(kd_loss)

    with open(args.output_file, 'a') as f:
        f.write(str(kd_loss_mean) + ',' + str(kd_loss_std) + ',')
        f.close()

def IS_FID_cal():
    gen_imgs_ = []
    true_images_ = []
    for i in range(args.number_classes):
        files0 = os.listdir(args.image_dir + args.dataset_teacher + '/real_images/' + str(i))
        files1 = os.listdir(args.image_dir + args.dataset_teacher + '/fake_images/' + str(i))
        # read 3-channel images
        img0 = [im2tensor(load_image(os.path.join(args.image_dir + args.dataset_teacher + '/real_images/' + str(i), file0))).cpu().numpy().reshape([3, args.img_size, args.img_size]) for file0 in files0[0:5]]  # RGB image from [-1,1]
        img1 = [im2tensor(load_image(os.path.join(args.image_dir + args.dataset_teacher + '/fake_images/' + str(i), file1))).cpu().numpy().reshape([3, args.img_size, args.img_size]) for file1 in files1[0:5]]
        true_images_.extend(img0)
        gen_imgs_.extend(img1)

    indices = np.array(range(len(true_images_)))
    np.random.shuffle(indices)
    true_images_ = np.array([true_images_[j] for j in indices])  # [0:opt.batch_size]
    indices = np.array(range(len(gen_imgs_)))
    np.random.shuffle(indices)
    gen_imgs_ = np.array([gen_imgs_[j] for j in indices])

    true_fake_images = []
    true_fake_images.append(true_images_)
    true_fake_images.append(gen_imgs_)

    print("Calculate IS & MID values...")

    if use_gpu:
        incept_score_mean, incept_score_std = inception_score(torch.tensor(gen_imgs_), cuda=True)
        fid_value = calculate_fid_given_images(true_fake_images, batch_size=50, cuda=True, dims=2048)
    else:
        incept_score_mean, incept_score_std = inception_score(torch.tensor(gen_imgs_), cuda=False)
        fid_value = calculate_fid_given_images(true_fake_images, batch_size=50, cuda=False, dims=2048)

    print(" [IS_mean: %f] [IS_std: %f] [FID: %f] " % (incept_score_mean, incept_score_std, fid_value))
    with open(args.output_file, 'a') as f:
        f.write(str(incept_score_mean) + ',' + str(fid_value) + ',')
        f.close()



with open(args.output_file, 'a') as f:
    f.write('Dataset'+','+'AccuracyTeacher'+','+'AccuracyStudent'+','+ 'LossKDmean'+','
            +'LossKDstd'+','+'IS'+','+'FID'+','+'LPIPS' + '\n')
    f.close()

get_teacher_model()
get_student_model()
get_dataset()
gen_images()
del generator
save_real_images()
save_fake_images()
del gen_imgs
del pred
test_teacher()
del teacher
test_student()
del student
del data_test_loader
del criterion
del data_test
test_results()
IS_FID_cal()
LPIPS_cal(args.dataset_teacher, args.number_classes, args.image_dir, args.output_file, device)



from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataloader import *
from nets.net_audiovisual import HCMN, AudioNet, VideoNet, DHHN
from utils.eval_metrics import segment_level, event_level
import pandas as pd
import pickle as pkl
from utils.logger import get_logger
import pdb
import time

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def retrain(args, model, train_loader, optimizer, criterion, epoch):

    co_model = model[0]
    audio_model = model[1]
    video_model = model[2]
    co_model.train()
    audio_model.train()
    video_model.train()

    for batch_idx, sample in enumerate(train_loader):
        audio, video, video_st, target = sample['audio'].to('cuda'), sample['video_s'].to('cuda'), sample['video_st'].to('cuda'), sample['label'].type(torch.FloatTensor).to('cuda')

        # optimizer.zero_grad()
        for optz in optimizer:
            optz.zero_grad()
        output, a_prob, v_prob, _, fa_aggr, fv = co_model(audio, video, video_st)
        output.clamp_(min=1e-7, max=1 - 1e-7)
        a_prob.clamp_(min=1e-7, max=1 - 1e-7)
        v_prob.clamp_(min=1e-7, max=1 - 1e-7)

        Pa = sample['pa'].type(torch.FloatTensor).to('cuda')
        Pv = sample['pv'].type(torch.FloatTensor).to('cuda')
        
        loss1 =  criterion(a_prob, Pa) 
        loss2 =  criterion(v_prob, Pv) 
        loss3 =  criterion(output, target) 
        loss = loss1 + loss2 + loss3

        audio_pred, _, _, _ = audio_model(audio, video, video_st)
        video_pred, _, _, _ = video_model(audio, video, video_st)
        audio_pred.clamp_(min=1e-7, max=1 - 1e-7)
        video_pred.clamp_(min=1e-7, max=1 - 1e-7)

        loss5 = criterion(audio_pred, Pa) 
        loss6 = criterion(video_pred, Pv)
        # loss7 = criterion3(audio_pred, video_pred)
        loss7 = -F.mse_loss(audio_pred, video_pred)
        loss8 = -F.mse_loss(a_prob, v_prob)
        loss_a_dis = F.mse_loss(a_prob, audio_pred.detach())
        loss_v_dis = F.mse_loss(v_prob, video_pred.detach())
        log_str = 'Train Epoch: {} [{}/{} ({:.0f}%)]\t audio Loss: {:.3f}\t video Loss: {:.3f}\t wsl Loss: {:.3f}\t dist_a: {:.3f}\t dist_v: {:.3f}, adv: {:.3f}; \n[SubNet]\t audio_net loss: {:.3f} \t video_net loss: {:.3f} \t adv loss: {:.3f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss1.item(),  loss2.item(), loss3.item(), loss_a_dis.item(), loss_v_dis.item(), loss8.item(), 
                loss5.item(), loss6.item(), loss7.item())
        loss = loss + loss_a_dis + loss_v_dis + loss8
        loss.backward(retain_graph=True)
        loss5.backward(retain_graph=True)
        loss6.backward(retain_graph=True)
        loss7.backward()
        for optz in optimizer:
            optz.step()
        if batch_idx % args.log_interval == 0:
            logger(log_str)


def train(args, model, train_loader, optimizer, criterion, epoch):

    co_model = model[0]
    audio_model = model[1]
    video_model = model[2]
    co_model.train()
    audio_model.train()
    video_model.train()

    for batch_idx, sample in enumerate(train_loader):
        audio, video, video_st, target = sample['audio'].to('cuda'), sample['video_s'].to('cuda'), sample['video_st'].to('cuda'), sample['label'].type(torch.FloatTensor).to('cuda')
        data_idx = sample['idx']
        # optimizer.zero_grad()
        for optz in optimizer:
            optz.zero_grad()
        output, a_prob, v_prob, _, _, _ = co_model(audio, video, video_st)
        output.clamp_(min=1e-7, max=1 - 1e-7)
        a_prob.clamp_(min=1e-7, max=1 - 1e-7)
        v_prob.clamp_(min=1e-7, max=1 - 1e-7)

        Pa = sample['pa'].type(torch.FloatTensor).to('cuda')
        Pv = sample['pv'].type(torch.FloatTensor).to('cuda')
        
        b=audio.size(0)
        loss1 =  criterion(a_prob, Pa) 
        loss2 =  criterion(v_prob, Pv) 
        loss3 =  criterion(output, target) 

        # loss4 = criterion2(fa_aggr, fv)
        # loss = loss1 + loss2 + loss3 + loss4 
        loss = loss1 + loss2 + loss3

        audio_pred, _, _, _ = audio_model(audio, video, video_st)
        video_pred, _, _, _ = video_model(audio, video, video_st)
        audio_pred.clamp_(min=1e-7, max=1 - 1e-7)
        video_pred.clamp_(min=1e-7, max=1 - 1e-7)

        loss5 = criterion(audio_pred, target) 
        loss6 = criterion(video_pred, target)
        # loss7 = criterion3(audio_pred, video_pred)
        loss7 = -F.mse_loss(audio_pred, video_pred)
        loss8 = -F.mse_loss(a_prob, v_prob)
        loss_a_dis = F.mse_loss(a_prob, audio_pred.detach())
        loss_v_dis = F.mse_loss(v_prob, video_pred.detach())
        log_str = 'Train Epoch: {} [{}/{} ({:.0f}%)]\t audio Loss: {:.3f}\t video Loss: {:.3f}\t wsl Loss: {:.3f}\t dist_a: {:.3f}\t dist_v: {:.3f}, adv: {:.3f}; \n[SubNet]\t audio_dis loss: {:.3f} \t video_dis loss: {:.3f} \t anti_dis loss: {:.3f}'.format(
                epoch, batch_idx * len(audio), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss1.item(),  loss2.item(), loss3.item(), loss_a_dis.item(), loss_v_dis.item(), loss8.item(), 
                loss5.item(), loss6.item(), loss7.item())
        loss = loss + loss_a_dis + loss_v_dis + loss8
        loss.backward(retain_graph=True)
        loss5.backward(retain_graph=True)
        loss6.backward(retain_graph=True)
        loss7.backward()
        for optz in optimizer:
            optz.step()
        if batch_idx % args.log_interval == 0:
            logger(log_str)



def eval(model, val_loader, set):
    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']
    # model.eval()
    for mol in model:             
        mol.eval()


    # load annotations
    df = pd.read_csv(set, header=0, sep='\t')
    df_a = pd.read_csv("data/AVVP_eval_audio.csv", header=0, sep='\t')
    df_v = pd.read_csv("data/AVVP_eval_visual.csv", header=0, sep='\t')

    id_to_idx = {id: index for index, id in enumerate(categories)}
    F_seg_a = []
    F_seg_v = []
    F_seg = []
    F_seg_av = []
    F_event_a = []
    F_event_v = []
    F_event = []
    F_event_av = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio, video, video_st, target = sample['audio'].to('cuda'), sample['video_s'].to('cuda'),sample['video_st'].to('cuda'), sample['label'].to('cuda')
            output, a_prob, v_prob, frame_prob, a, is_real = model[0](audio, video, video_st)
            o = (output.cpu().detach().numpy() >= 0.5).astype(np.int_)
            oa = (a_prob.cpu().detach().numpy() >= 0.5).astype(np.int_)
            ov = (v_prob.cpu().detach().numpy() >= 0.5).astype(np.int_)
            
            Pa = frame_prob[0, :, 0, :].cpu().detach().numpy()
            Pv = frame_prob[0, :, 1, :].cpu().detach().numpy()

            Pa = (Pa >= 0.5).astype(np.int_) * np.repeat(oa, repeats=10, axis=0)
            Pv = (Pv >= 0.5).astype(np.int_) * np.repeat(ov, repeats=10, axis=0)

            # extract audio GT labels
            GT_a = np.zeros((25, 10))
            GT_v =np.zeros((25, 10))
            GT_aa = np.zeros(25, dtype=np.int)
            GT_vv = np.zeros(25, dtype=np.int)

            df_vid_a = df_a.loc[df_a['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_a["filename"]
            events = df_vid_a["event_labels"]
            onsets = df_vid_a["onset"]
            offsets = df_vid_a["offset"]
            num = len(filenames)
            if num >0:
                for i in range(num):

                    x1 = int(onsets[df_vid_a.index[i]])
                    x2 = int(offsets[df_vid_a.index[i]])
                    event = events[df_vid_a.index[i]]
                    idx = id_to_idx[event]
                    GT_a[idx, x1:x2] = 1
                    GT_aa[idx] = 1

            # extract visual GT labels
            df_vid_v = df_v.loc[df_v['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_v["filename"]
            events = df_vid_v["event_labels"]
            onsets = df_vid_v["onset"]
            offsets = df_vid_v["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_v.index[i]])
                    x2 = int(offsets[df_vid_v.index[i]])
                    event = events[df_vid_v.index[i]]
                    idx = id_to_idx[event]
                    GT_v[idx, x1:x2] = 1
                    GT_vv[idx]=1
            GT_av = GT_a * GT_v

            # obtain prediction matrices
            SO_a = np.transpose(Pa)
            SO_v = np.transpose(Pv)
            SO_av = SO_a * SO_v

            # segment-level F1 scores
            f_a, f_v, f, f_av = segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_seg_a.append(f_a)
            F_seg_v.append(f_v)
            F_seg.append(f)
            F_seg_av.append(f_av)

            # event-level F1 scores
            f_a, f_v, f, f_av = event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_event_a.append(f_a)
            F_event_v.append(f_v)
            F_event.append(f)
            F_event_av.append(f_av)

    logger("\n")    
    logger('Audio  \t {:.1f} \t {:.1f}'.format( 100 * np.mean(np.array(F_seg_a)),  100 * np.mean(np.array(F_event_a))))
    logger('Visual \t {:.1f} \t {:.1f}'.format( 100 * np.mean(np.array(F_seg_v)),  100 * np.mean(np.array(F_event_v))))
    logger('AudVis \t {:.1f} \t {:.1f}'.format( 100 * np.mean(np.array(F_seg_av)),  100 * np.mean(np.array(F_event_av))))

    audio_score = (100 * np.mean(np.array(F_seg_a)) + 100 * np.mean(np.array(F_event_a)))/2.0
    video_score = (100 * np.mean(np.array(F_seg_v)) + 100 * np.mean(np.array(F_event_v)))/2.0


    avg_type = (100 * np.mean(np.array(F_seg_av))+100 * np.mean(np.array(F_seg_a))+100 * np.mean(np.array(F_seg_v)))/3.
    avg_event = 100 * np.mean(np.array(F_seg))
    logger('Segment-levelType@Avg. F1: {:.1f}'.format(avg_type))
    logger('Segment-level Event@Avg. F1: {:.1f}'.format( avg_event))

    avg_type_event = (100 * np.mean(np.array(F_event_av)) + 100 * np.mean(np.array(F_event_a)) + 100 * np.mean(
        np.array(F_event_v))) / 3.
    avg_event_level = 100 * np.mean(np.array(F_event))
    logger('Event-level Type@Avg. F1: {:.1f}'.format( avg_type_event))
    logger('Event-level Event@Avg. F1: {:.1f}'.format(avg_event_level))
    logger("\n")
    return avg_type, audio_score, video_score


def final_eval(model, val_loader, set):
    categories = ['Speech', 'Car', 'Cheering', 'Dog', 'Cat', 'Frying_(food)',
                  'Basketball_bounce', 'Fire_alarm', 'Chainsaw', 'Cello', 'Banjo',
                  'Singing', 'Chicken_rooster', 'Violin_fiddle', 'Vacuum_cleaner',
                  'Baby_laughter', 'Accordion', 'Lawn_mower', 'Motorcycle', 'Helicopter',
                  'Acoustic_guitar', 'Telephone_bell_ringing', 'Baby_cry_infant_cry', 'Blender',
                  'Clapping']
    # model.eval()
    model.eval()

    # load annotations
    df = pd.read_csv(set, header=0, sep='\t')
    df_a = pd.read_csv("data/AVVP_eval_audio.csv", header=0, sep='\t')
    df_v = pd.read_csv("data/AVVP_eval_visual.csv", header=0, sep='\t')

    id_to_idx = {id: index for index, id in enumerate(categories)}
    F_seg_a = []
    F_seg_v = []
    F_seg = []
    F_seg_av = []
    F_event_a = []
    F_event_v = []
    F_event = []
    F_event_av = []

    with torch.no_grad():
        for batch_idx, sample in enumerate(val_loader):
            audio, video, video_st, target = sample['audio'].to('cuda'), sample['video_s'].to('cuda'),sample['video_st'].to('cuda'), sample['label'].to('cuda')
            a_prob, a_frame_prob, v_prob, v_frame_prob = model(audio, video, video_st)
            oa = (a_prob.cpu().detach().numpy() >= 0.5).astype(np.int_)
            ov = (v_prob.cpu().detach().numpy() >= 0.5).astype(np.int_)
            
            Pa = a_frame_prob[0, :, 0, :].cpu().detach().numpy()
            Pv = v_frame_prob[0, :, 1, :].cpu().detach().numpy()

            Pa = (Pa >= 0.5).astype(np.int_) * np.repeat(oa, repeats=10, axis=0)
            Pv = (Pv >= 0.5).astype(np.int_) * np.repeat(ov, repeats=10, axis=0)

            # extract audio GT labels
            GT_a = np.zeros((25, 10))
            GT_v =np.zeros((25, 10))
            GT_aa = np.zeros(25, dtype=np.int)
            GT_vv = np.zeros(25, dtype=np.int)

            df_vid_a = df_a.loc[df_a['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_a["filename"]
            events = df_vid_a["event_labels"]
            onsets = df_vid_a["onset"]
            offsets = df_vid_a["offset"]
            num = len(filenames)
            if num >0:
                for i in range(num):

                    x1 = int(onsets[df_vid_a.index[i]])
                    x2 = int(offsets[df_vid_a.index[i]])
                    event = events[df_vid_a.index[i]]
                    idx = id_to_idx[event]
                    GT_a[idx, x1:x2] = 1
                    GT_aa[idx] = 1

            # extract visual GT labels
            df_vid_v = df_v.loc[df_v['filename'] == df.loc[batch_idx, :][0]]
            filenames = df_vid_v["filename"]
            events = df_vid_v["event_labels"]
            onsets = df_vid_v["onset"]
            offsets = df_vid_v["offset"]
            num = len(filenames)
            if num > 0:
                for i in range(num):
                    x1 = int(onsets[df_vid_v.index[i]])
                    x2 = int(offsets[df_vid_v.index[i]])
                    event = events[df_vid_v.index[i]]
                    idx = id_to_idx[event]
                    GT_v[idx, x1:x2] = 1
                    GT_vv[idx]=1
            GT_av = GT_a * GT_v

            # obtain prediction matrices
            SO_a = np.transpose(Pa)
            SO_v = np.transpose(Pv)
            SO_av = SO_a * SO_v

            # segment-level F1 scores
            f_a, f_v, f, f_av = segment_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_seg_a.append(f_a)
            F_seg_v.append(f_v)
            F_seg.append(f)
            F_seg_av.append(f_av)

            # event-level F1 scores
            f_a, f_v, f, f_av = event_level(SO_a, SO_v, SO_av, GT_a, GT_v, GT_av)
            F_event_a.append(f_a)
            F_event_v.append(f_v)
            F_event.append(f)
            F_event_av.append(f_av)

    logger("\n")    
    logger('Audio  \t {:.1f} \t {:.1f}'.format( 100 * np.mean(np.array(F_seg_a)),  100 * np.mean(np.array(F_event_a))))
    logger('Visual \t {:.1f} \t {:.1f}'.format( 100 * np.mean(np.array(F_seg_v)),  100 * np.mean(np.array(F_event_v))))
    logger('AudVis \t {:.1f} \t {:.1f}'.format( 100 * np.mean(np.array(F_seg_av)),  100 * np.mean(np.array(F_event_av))))

    audio_score = (100 * np.mean(np.array(F_seg_a)) + 100 * np.mean(np.array(F_event_a)))/2.0
    video_score = (100 * np.mean(np.array(F_seg_v)) + 100 * np.mean(np.array(F_event_v)))/2.0


    avg_type = (100 * np.mean(np.array(F_seg_av))+100 * np.mean(np.array(F_seg_a))+100 * np.mean(np.array(F_seg_v)))/3.
    avg_event = 100 * np.mean(np.array(F_seg))
    logger('Segment-levelType@Avg. F1: {:.1f}'.format(avg_type))
    logger('Segment-level Event@Avg. F1: {:.1f}'.format( avg_event))

    avg_type_event = (100 * np.mean(np.array(F_event_av)) + 100 * np.mean(np.array(F_event_a)) + 100 * np.mean(
        np.array(F_event_v))) / 3.
    avg_event_level = 100 * np.mean(np.array(F_event))
    logger('Event-level Type@Avg. F1: {:.1f}'.format( avg_type_event))
    logger('Event-level Event@Avg. F1: {:.1f}'.format(avg_event_level))
    logger("\n")
    return avg_type, audio_score, video_score


logger = get_logger()

def main():
    parser = argparse.ArgumentParser(description='PyTorch Implementation of Audio-Visual Video Parsing')
    parser.add_argument(
        "--audio_dir", type=str, default='data/feats/vggish/', help="audio dir")
    parser.add_argument(
        "--video_dir", type=str, default='data/feats/res152/',
        help="video dir")
    parser.add_argument(
        "--st_dir", type=str, default='data/feats/r2plus1d_18/',
        help="video dir")
    parser.add_argument(
        "--label_train", type=str, default="data/AVVP_train.csv", help="weak train csv file")
    parser.add_argument(
        "--label_val", type=str, default="data/AVVP_val_pd.csv", help="weak val csv file")
    parser.add_argument(
        "--label_test", type=str, default="data/AVVP_test_pd.csv", help="weak test csv file")
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 60)')
    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',
                        help='learning rate (default: 3e-4)')
    parser.add_argument(
        "--mode", type=str, default='train', help="with mode to use")
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument(
        "--model_save_dir", type=str, default='models/', help="model save dir")
    parser.add_argument(
        "--checkpoint", type=str, default='DHHN',
        help="save model name")
    parser.add_argument(
        '--gpu', type=str, default='0', help='gpu device number')
    parser.add_argument(
        "--logger", type=str, default='logger/',
        help="save model name")
    parser.add_argument(
        "--no-log", action='store_true', default=False,
        help="logger switcher")
    args = parser.parse_args()

    setup_seed(args.seed)

    if args.no_log:
        logger.disable_file()
    else:
        logger.set_file(os.path.join(args.logger, args.checkpoint+".txt"))

    model = [HCMN().to('cuda'), AudioNet().to('cuda'), VideoNet().to('cuda')]


    if args.mode == 'retrain':
        # start_time = time.time()
        train_dataset = LLP_dataset(label=args.label_train, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, mode='retrain', transform = transforms.Compose([ToTensor()]))
        val_dataset = LLP_dataset(label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, mode='val', transform = transforms.Compose([ToTensor()]))
        test_dataset = LLP_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir,
            st_dir=args.st_dir, mode='test', transform=transforms.Compose([ToTensor()]))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory = True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory = True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        optimizer = [optim.Adam(model[0].parameters(), lr=args.lr), optim.Adam(model[1].parameters(), lr=args.lr), optim.Adam(model[2].parameters(), lr=args.lr)]
        scheduler = [optim.lr_scheduler.StepLR(optz, step_size=10, gamma=0.1) for optz in optimizer]
        criterion = nn.BCELoss()
        best_F = 0
        best_audio = 0
        best_video = 0
 
        for epoch in range(1, args.epochs + 1):
            retrain(args, model, train_loader, optimizer, criterion, epoch=epoch)
            for sch in scheduler:
                sch.step()
            print("Validation Performance of Epoch {}:".format(epoch))
            # F = eval(model, val_loader, args.label_val)
            F, audio, video = eval(model, val_loader, args.label_val)
            if F > best_F:
                best_F = F
                logger("==> best_F [{}] at [{}]".format(best_F, epoch))
                logger("==> Save best audio-visual checkpoint at {}.\n".format(args.model_save_dir + args.checkpoint + "_aud-vis.pt"))
                torch.save(model[0].state_dict(), args.model_save_dir + args.checkpoint + "_aud-vis.pt")
            if audio > best_audio:
                best_audio = audio
                logger("==> best_audio [{}] at [{}]".format(best_audio, epoch))
                logger("==> Save best audio checkpoint at {}.\n".format(args.model_save_dir + args.checkpoint + "_audio.pt"))
                torch.save(model[0].state_dict(), args.model_save_dir + args.checkpoint + "_audio.pt")
            if video > best_video:
                best_video = video
                logger("==> best_video [{}] at [{}]".format(best_video, epoch))
                logger("==> Save best video checkpoint at {}.\n".format(args.model_save_dir + args.checkpoint + "_video.pt"))
                torch.save(model[0].state_dict(), args.model_save_dir + args.checkpoint + "_video.pt")

    elif args.mode == 'train':
        start_time = time.time()
        train_dataset = LLP_dataset(label=args.label_train, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, mode='train', transform = transforms.Compose([ToTensor()]))
        val_dataset = LLP_dataset(label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir, st_dir=args.st_dir, mode='val', transform = transforms.Compose([ToTensor()]))
        test_dataset = LLP_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir,
            st_dir=args.st_dir, mode='test', transform=transforms.Compose([ToTensor()]))

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory = True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory = True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

        optimizer = [optim.Adam(model[0].parameters(), lr=args.lr), optim.Adam(model[1].parameters(), lr=2.0*args.lr), optim.Adam(model[2].parameters(), lr=0.2*args.lr)]
        scheduler = [optim.lr_scheduler.StepLR(optz, step_size=10, gamma=0.1) for optz in optimizer]
        criterion = nn.BCELoss()
        best_F = 0
        best_audio = 0
        best_video = 0
 
        for epoch in range(1, args.epochs + 1):
            train(args, model, train_loader, optimizer, criterion, epoch=epoch)
            for sch in scheduler:
                sch.step()
            F, audio, video = eval(model, val_loader, args.label_val)
            if F > best_F:
                best_F = F
                logger("==> best_F [{}] at [{}]".format(best_F, epoch))
                logger("==> Save best audio-visual checkpoint at {}.\n".format(args.model_save_dir + args.checkpoint + "_aud-vis.pt"))
                torch.save(model[0].state_dict(), args.model_save_dir + args.checkpoint + "_aud-vis.pt")
            if audio > best_audio:
                best_audio = audio
                logger("==> best_audio [{}] at [{}]".format(best_audio, epoch))
                logger("==> Save best audio checkpoint at {}.\n".format(args.model_save_dir + args.checkpoint + "_audio.pt"))
                torch.save(model[0].state_dict(), args.model_save_dir + args.checkpoint + "_audio.pt")
            if video > best_video:
                best_video = video
                logger("==> best_video [{}] at [{}]".format(best_video, epoch))
                logger("==> Save best video checkpoint at {}.\n".format(args.model_save_dir + args.checkpoint + "_video.pt"))
                torch.save(model[0].state_dict(), args.model_save_dir + args.checkpoint + "_video.pt")
        end_time = time.time()
        logger("Toal training time(S): ",end_time - start_time)

    elif args.mode == 'val':
        test_dataset = LLP_dataset(label=args.label_val, audio_dir=args.audio_dir, video_dir=args.video_dir,
                                    st_dir=args.st_dir, mode='val', transform=transforms.Compose([
                ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        # model[0].load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
        # eval(model, test_loader, args.label_val)
        final_model = DHHN().to('cuda')
        final_model.audio_net.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + "_audio.pt"))
        final_model.video_net.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + "_video.pt"))
        final_eval(final_model, test_loader, args.label_val)
    
    else:
        logger("Testing...")
        test_dataset = LLP_dataset(label=args.label_test, audio_dir=args.audio_dir, video_dir=args.video_dir,  st_dir=args.st_dir, mode='test', transform = transforms.Compose([
                                               ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
        # model[0].load_state_dict(torch.load(args.model_save_dir + args.checkpoint + ".pt"))
        # eval(model, test_loader, args.label_test)
        final_model = DHHN().to('cuda')
        final_model.audio_net.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + "_audio.pt"))
        final_model.video_net.load_state_dict(torch.load(args.model_save_dir + args.checkpoint + "_video.pt"))
        final_eval(final_model, test_loader, args.label_test)

if __name__ == '__main__':
    main()

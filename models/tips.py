import hashlib
import logging
import os
import urllib
import warnings

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.optim import Optimizer
import math
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import TipsVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, count_parameters
#import clip
from CLIP import CLIP, CustomCLIP, clip_build_model

# tune the model at first session with vpt, and then conduct simple shot.
num_workers = 8
networkmodel_network ='network_save/networkvit.pth'
clipmodel_network ='network_save/clipmodel.pth'


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

       
        cilpmodel = self.load_clip_to_cpu()
        classorder=args["classorder"]
        classnamess1 = get_cub200_class_names()
        classnames = [None] * len(classnamess1)
        for index, value in enumerate(classorder):
            classnames[index] = classnamess1[value]
        inc_cls =args['increment']
        self.classname_origin = classnamess1
        self.classname_order = classnames
        self.clipmodel = CustomCLIP(cfg=True, classnames=classnames, inc_class = inc_cls,clip_model=cilpmodel).to(self._device)
        for name, param in self.clipmodel.named_parameters():
            if "prompt_learner" not in name and "logit" not in name:
                #print(name)
                param.requires_grad_(False)
        print(("cilpmodel Trainable params: {}".format(count_parameters(self.clipmodel, True))))
        self._network = TipsVitNet(args,True).to(self._device)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args

        total_params = sum(p.numel() for p in self._network.parameters())
        logging.info(f'{total_params:,} total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.fc.parameters() if p.requires_grad) + sum(
            p.numel() for p in self._network.prompt.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} fc and prompt training parameters.')

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1

        if self._cur_task > 0:
            try:
                if self._network.module.prompt is not None:
                    self._network.module.prompt.process_task_count()
            except:
                if self._network.prompt is not None:
                    self._network.prompt.process_task_count()

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        # self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(np.arange(self._known_classes, self._total_classes), source="train",
                                                 mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True,
                                       num_workers=num_workers)
        test_dataset = data_manager.get_dataset(np.arange(0, self._total_classes), source="test", mode="test")
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False,
                                      num_workers=num_workers)

        if len(self._multiple_gpus) > 1:
            print('Multiple GPUs')
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
        self._train(self.train_loader, self.test_loader)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)

        optimizer = self.get_optimizer()
        scheduler = self.get_scheduler(optimizer)

        self.data_weighting()
        self._init_train(train_loader, test_loader, optimizer, scheduler)

    def data_weighting(self):
        self.dw_k = torch.tensor(np.ones(self._total_classes + 1, dtype=np.float32))
        self.dw_k = self.dw_k.to(self._device)

    def get_optimizer(self):
        if len(self._multiple_gpus) > 1:
            params = list(self._network.module.prompt.parameters()) + list(self._network.module.fc.parameters())
        else:
            params = list(self._network.prompt.parameters()) + list(self._network.fc.parameters())
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(params, momentum=0.9, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(params, lr=self.init_lr, weight_decay=self.weight_decay)
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(params, lr=self.init_lr, weight_decay=self.weight_decay)

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = CosineSchedule(optimizer, K=self.args["tuned_epoch"])
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[10,20,30,40,50,60],
                                                       gamma=0.5)
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def get_image_embedding(self, images, model):
        # image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features


    def convert_weights(self,model: nn.Module):
        """Convert applicable model parameters to fp16"""

        def _convert_weights_to_fp16(l):
            if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                l.weight.data = l.weight.data.half()
                if l.bias is not None:
                    l.bias.data = l.bias.data.half()

            if isinstance(l, nn.MultiheadAttention):
                for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                    tensor = getattr(l, attr)
                    if tensor is not None:
                        tensor.data = tensor.data.half()

            for name in ["text_projection", "proj"]:
                if hasattr(l, name):
                    attr = getattr(l, name)
                    if attr is not None:
                        attr.data = attr.data.half()

        model.apply(_convert_weights_to_fp16)

    def load_clip_to_cpu(self):
        #url = "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"
        url ="https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"
        model_path = self.clip_download(url)


        try:
            # loading JIT archive
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None

        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        model = clip_build_model(state_dict or model.state_dict())

        return model


    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        #dict1 = self.cifar100_label_to_name_dictionary()

        #print(cilpmodel)
        # for param in cilpmodel.parameters():
        #     param.requires_grad = False
        # print(("cilpmodel Trainable params: {}".format(count_parameters(cilpmodel, True))))
        best_acc =0.0
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            losses_kd =0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                text_features,image_features,clip_logits= self.clipmodel(inputs,targets,self._cur_task,train=True)
                if clip_logits is not None:
                    pass
                else:
                    print("tips init bug")
                # logits
                logits, lang , prompt_loss = self._network(inputs, query=image_features,sim_logit=clip_logits, train=True)
                logits = logits[:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')
                dw_cls = self.dw_k[-1 * torch.ones(targets.size()).long()]
                loss_supervised = (F.cross_entropy(logits, targets.long()) * dw_cls).mean()

                loss_kd =_KD_loss(lang, text_features, 2).mean() 

                # ce loss
                loss = loss_supervised + prompt_loss.sum()  + loss_kd

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses += loss.item()
                losses_kd += loss_kd.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()

            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            if (epoch + 1) % 1 == 0:
                test_acc = self._compute_accuracy(test_loader)  
                if test_acc > best_acc:
                    best_acc = test_acc
                    torch.save(self._network.state_dict(), networkmodel_network)
                    torch.save(self.clipmodel.state_dict(), clipmodel_network)
                    print("Epoch {} => best_acc {:.2f}".format(epoch + 1, best_acc))
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, where lang_KDLoss {:.3f},Train_accy {:.2f}, Test_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    losses_kd / len(train_loader),
                    train_acc,
                    test_acc,
                )
            else:
                info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                    self._cur_task,
                    epoch + 1,
                    self.args['tuned_epoch'],
                    losses / len(train_loader),
                    train_acc,
                )
            prog_bar.set_description(info)
        #self._network.load_state_dict(torch.load(networkmodel_network))
        #self.clipmodel.load_state_dict(torch.load(clipmodel_network))
        #self.clipmodel.eval()
        #self._network.eval()
        logging.info(info)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                text_features, image_features, clip_logits = self.clipmodel(inputs, targets, self._cur_task,
                                                                            train=False)
                outputs = self._network(inputs, query=image_features, sim_logit=clip_logits, train=False)[:,
                          :self._total_classes]
                # outputs = self._network(inputs)[:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _compute_accuracy(self,loader):
        self._network.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                text_features, image_features, clip_logits = self.clipmodel(inputs, targets, self._cur_task, train=False)
                outputs = self._network(inputs, query=image_features,sim_logit=clip_logits, train=False)[:, :self._total_classes]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def cifar100_label_to_name_dictionary(self):
        label_name_dict = {
            0: "apple", 1: "aquarium_fish", 2: "baby", 3: "bear", 4: "beaver",
            5: "bed", 6: "bee", 7: "beetle", 8: "bicycle", 9: "bottle",
            10: "bowl", 11: "boy", 12: "bridge", 13: "bus", 14: "butterfly",
            15: "camel", 16: "can", 17: "castle", 18: "caterpillar", 19: "cattle",
            20: "chair", 21: "chimpanzee", 22: "clock", 23: "cloud", 24: "cockroach",
            25: "couch", 26: "crab", 27: "crocodile", 28: "cup", 29: "dinosaur",
            30: "dolphin", 31: "elephant", 32: "flatfish", 33: "forest", 34: "fox",
            35: "girl", 36: "hamster", 37: "house", 38: "kangaroo", 39: "keyboard",
            40: "lamp", 41: "lawn_mower", 42: "leopard", 43: "lion", 44: "lizard",
            45: "lobster", 46: "man", 47: "maple_tree", 48: "motorcycle", 49: "mountain",
            50: "mouse", 51: "mushrooms", 52: "oak_tree", 53: "orange", 54: "orchids",
            55: "otter", 56: "palm_tree", 57: "pears", 58: "pickup_truck", 59: "pine_tree",
            60: "plain", 61: "plate", 62: "poppy", 63: "porcupine", 64: "possum",
            65: "rabbit", 66: "raccoon", 67: "ray", 68: "road", 69: "rocket",
            70: "roses", 71: "sea", 72: "seal", 73: "shark", 74: "shrew",
            75: "skunk", 76: "skyscraper", 77: "snail", 78: "snake", 79: "spider",
            80: "squirrel", 81: "streetcar", 82: "sunflowers", 83: "sweet_peppers", 84: "table",
            85: "tank", 86: "telephone", 87: "television", 88: "tiger", 89: "tractor",
            90: "train", 91: "trout", 92: "tulips", 93: "turtle", 94: "wardrobe",
            95: "whale", 96: "willow_tree", 97: "wolf", 98: "woman", 99: "worm",
        }
        return label_name_dict

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy


def get_cifar100_classes():
    return [
        'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle',
        'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel',
        'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock',
        'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur',
        'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster',
        'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion',
        'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse',
        'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear',
        'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
        'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
        'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
        'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table',
        'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
        'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
        'worm'
    ]


def get_cub200_class_names():
    class_names = [
        'Black_footed_Albatross', 'Laysan_Albatross', 'Sooty_Albatross', 'Groove_billed_Ani', 'Crested_Auklet',
        'Least_Auklet', 'Parakeet_Auklet', 'Rhinoceros_Auklet', 'Brewer_Blackbird', 'Red_winged_Blackbird',
        'Rusty_Blackbird', 'Yellow_headed_Blackbird', 'Bobolink', 'Indigo_Bunting', 'Lazuli_Bunting', 'Painted_Bunting',
        'Cardinal', 'Spotted_Catbird', 'Gray_Catbird', 'Yellow_breasted_Chat', 'Eastern_Towhee', 'Chuck_will_Widow',
        'Brandt_Cormorant', 'Red_faced_Cormorant', 'Pelagic_Cormorant', 'Bronzed_Cowbird', 'Shiny_Cowbird',
        'Brown_Creeper', 'American_Crow', 'Fish_Crow', 'Black_billed_Cuckoo', 'Mangrove_Cuckoo', 'Yellow_billed_Cuckoo',
        'Gray_crowned_Rosy_Finch', 'Purple_Finch', 'Northern_Flicker', 'Acadian_Flycatcher', 'Great_Crested_Flycatcher',
        'Least_Flycatcher', 'Olive_sided_Flycatcher', 'Scissor_tailed_Flycatcher', 'Vermilion_Flycatcher',
        'Yellow_bellied_Flycatcher', 'Frigatebird', 'Northern_Fulmar', 'Gadwall', 'American_Goldfinch',
        'European_Goldfinch', 'Boat_tailed_Grackle', 'Eared_Grebe', 'Horned_Grebe', 'Pied_billed_Grebe',
        'Western_Grebe', 'Blue_Grosbeak', 'Evening_Grosbeak', 'Pine_Grosbeak', 'Rose_breasted_Grosbeak',
        'Pigeon_Guillemot', 'California_Gull', 'Glaucous_winged_Gull', 'Heermann_Gull', 'Herring_Gull', 'Ivory_Gull',
        'Ring_billed_Gull', 'Slaty_backed_Gull', 'Western_Gull', 'Anna_Hummingbird', 'Ruby_throated_Hummingbird',
        'Rufous_Hummingbird', 'Green_Violetear', 'Long_tailed_Jaeger', 'Pomarine_Jaeger', 'Blue_Jay', 'Florida_Jay',
        'Green_Jay', 'Dark_eyed_Junco', 'Tropical_Kingbird', 'Gray_Kingbird', 'Belted_Kingfisher', 'Green_Kingfisher',
        'Pied_Kingfisher', 'Ringed_Kingfisher', 'White_breasted_Kingfisher', 'Red_legged_Kittiwake', 'Horned_Lark',
        'Pacific_Loon', 'Mallard', 'Western_Meadowlark', 'Hooded_Merganser', 'Red_breasted_Merganser', 'Mockingbird',
        'Nighthawk', 'Clark_Nutcracker', 'White_breasted_Nuthatch', 'Baltimore_Oriole', 'Hooded_Oriole', 'Orchard_Oriole',
        'Scott_Oriole', 'Ovenbird', 'Brown_Pelican', 'White_Pelican', 'Western_Wood_Pewee', 'Sayornis', 'American_Pipit',
        'Whip_poor_Will', 'Horned_Puffin', 'Common_Raven', 'White_necked_Raven', 'American_Redstart', 'Geococcyx',
        'Loggerhead_Shrike', 'Great_Grey_Shrike', 'Baird_Sparrow', 'Black_throated_Sparrow', 'Brewer_Sparrow',
        'Chipping_Sparrow', 'Clay_colored_Sparrow', 'House_Sparrow', 'Field_Sparrow', 'Fox_Sparrow', 'Grasshopper_Sparrow',
        'Harris_Sparrow', 'Henslow_Sparrow', 'Le_Conte_Sparrow', 'Lincoln_Sparrow', 'Nelson_Sharp_tailed_Sparrow',
        'Savannah_Sparrow', 'Seaside_Sparrow', 'Song_Sparrow', 'Tree_Sparrow', 'Vesper_Sparrow', 'White_crowned_Sparrow',
        'White_throated_Sparrow', 'Cape_Glossy_Starling', 'Bank_Swallow', 'Barn_Swallow', 'Cliff_Swallow', 'Tree_Swallow',
        'Scarlet_Tanager', 'Summer_Tanager', 'Artic_Tern', 'Black_Tern', 'Caspian_Tern', 'Common_Tern', 'Elegant_Tern',
        'Forsters_Tern', 'Least_Tern', 'Green_tailed_Towhee', 'Brown_Thrasher', 'Sage_Thrasher', 'Black_capped_Vireo',
        'Blue_headed_Vireo', 'Philadelphia_Vireo', 'Red_eyed_Vireo', 'Warbling_Vireo', 'White_eyed_Vireo',
        'Yellow_throated_Vireo', 'Bay_breasted_Warbler', 'Black_and_white_Warbler', 'Black_throated_Blue_Warbler',
        'Blue_winged_Warbler', 'Canada_Warbler', 'Cape_May_Warbler', 'Cerulean_Warbler', 'Chestnut_sided_Warbler',
        'Golden_winged_Warbler', 'Hooded_Warbler', 'Kentucky_Warbler', 'Magnolia_Warbler', 'Mourning_Warbler',
        'Myrtle_Warbler', 'Nashville_Warbler', 'Orange_crowned_Warbler', 'Palm_Warbler', 'Pine_Warbler',
        'Prairie_Warbler', 'Prothonotary_Warbler', 'Swainson_Warbler', 'Tennessee_Warbler', 'Wilson_Warbler',
        'Worm_eating_Warbler', 'Yellow_Warbler', 'Northern_Waterthrush', 'Louisiana_Waterthrush', 'Bohemian_Waxwing',
        'Cedar_Waxwing', 'American_Three_toed_Woodpecker', 'Pileated_Woodpecker', 'Red_bellied_Woodpecker',
        'Red_cockaded_Woodpecker', 'Red_headed_Woodpecker', 'Downy_Woodpecker', 'Bewick_Wren', 'Cactus_Wren',
        'Carolina_Wren', 'House_Wren', 'Marsh_Wren', 'Rock_Wren', 'Winter_Wren', 'Common_Yellowthroat'
    ]
    return class_names

def get_idomainnet_class_names():
    class_names = [
        'pickup_truck', 'sweater', 'The_Eiffel_Tower', 'backpack', 'waterslide', 'lion', 'submarine', 'toothpaste',
        'banana', 'television', 'cruise_ship', 'lighthouse', 'sandwich', 'bee', 'rhinoceros', 'tornado', 'golf_club',
        'ladder', 'tiger', 'castle', 'kangaroo', 'frog', 'pillow', 'goatee', 'horse', 'circle', 'toothbrush',
        'washing_machine', 'eye', 'river', 'basket', 'spoon', 'sea_turtle', 'bracelet', 'coffee_cup', 'strawberry',
        'squirrel', 'cat', 'teapot', 'mug', 'elephant', 'swan', 'raccoon', 'onion', 'animal_migration', 'nail',
        'sock', 'sword', 'hedgehog', 'donut', 'crocodile', 'ocean', 'leg', 'drums', 'hand', 'penguin', 'speedboat',
        'zigzag', 'pizza', 'windmill', 'square', 'shovel', 'guitar', 'sheep', 'suitcase', 'flashlight', 'steak',
        'duck', 'sailboat', 'triangle', 'yoga', 'broccoli', 'beach', 'fence', 'binoculars', 'garden', 'shorts',
        'mushroom', 'teddy-bear', 'bicycle', 'snowman', 'see_saw', 'pineapple', 'truck', 'shoe', 'nose', 'hexagon',
        'squiggle', 'umbrella', 'ice_cream', 'candle', 'grass', 'necklace', 'barn', 'popsicle', 'palm_tree',
        'rabbit', 'eyeglasses', 'motorbike', 'dog', 'bus', 'dumbbell', 'pond', 'whale', 'cake', 'wine_bottle',
        'moon', 'pig', 'sleeping_bag', 'feather', 'tent', 'brain', 'fish', 'bird', 'scorpion', 'wine_glass',
        'snorkel', 'map', 'stove', 'school_bus', 'pear', 'dolphin', 'rollerskates', 'moustache', 'mountain',
        'stethoscope', 'spider', 'bottlecap', 'bowtie', 'octopus', 'watermelon', 'car', 'toilet', 'light_bulb',
        'harp', 'door', 'asparagus', 'tree', 'table', 'toaster', 'tractor', 'spreadsheet', 'skateboard', 'helmet',
        'beard', 'monkey', 'vase', 'crab', 'skyscraper', 'snail', 'train', 'giraffe', 'finger', 'cup', 'butterfly',
        'scissors', 'zebra', 'parachute', 'stairs', 'bench', 'owl', 'book', 'rifle', 'carrot', 'trombone', 'flower',
        'streetlight', 'wristwatch', 'snake', 'pool', 'saxophone', 'grapes', 'violin', 'hamburger', 'bread', 'face',
        'bridge', 'hat', 'lobster', 'shark', 'headphones', 'leaf', 'house_plant', 'skull', 'telephone', 'envelope',
        'broom', 'hot_tub', 'bed', 'canoe', 'firetruck', 'bear', 'sun', 'syringe', 'hot_air_balloon', 'van',
        'helicopter', 'flamingo', 'blueberry', 'parrot'
    ]
    return class_names


def get_imagenetr_class_names():
    class_names = [
        'goldfish', 'great_white_shark', 'hammerhead', 'stingray', 'hen', 'ostrich', 'goldfinch', 'junco',
        'bald_eagle', 'vulture', 'newt', 'axolotl', 'tree_frog', 'iguana', 'African_chameleon', 'cobra', 'scorpion',
        'tarantula', 'centipede', 'peacock', 'lorikeet', 'hummingbird', 'toucan', 'duck', 'goose', 'black_swan', 'koala',
        'jellyfish', 'snail', 'lobster', 'hermit_crab', 'flamingo', 'american_egret', 'pelican', 'king_penguin',
        'grey_whale', 'killer_whale', 'sea_lion', 'chihuahua', 'shih_tzu', 'afghan_hound', 'basset_hound', 'beagle',
        'bloodhound', 'italian_greyhound', 'whippet', 'weimaraner', 'yorkshire_terrier', 'boston_terrier',
        'scottish_terrier', 'west_highland_white_terrier', 'golden_retriever', 'labrador_retriever', 'cocker_spaniels',
        'collie', 'border_collie', 'rottweiler', 'german_shepherd_dog', 'boxer', 'french_bulldog', 'saint_bernard',
        'husky', 'dalmatian', 'pug', 'pomeranian', 'chow_chow', 'pembroke_welsh_corgi', 'toy_poodle', 'standard_poodle',
        'timber_wolf', 'hyena', 'red_fox', 'tabby_cat', 'leopard', 'snow_leopard', 'lion', 'tiger', 'cheetah',
        'polar_bear', 'meerkat', 'ladybug', 'fly', 'bee', 'ant', 'grasshopper', 'cockroach', 'mantis', 'dragonfly',
        'monarch_butterfly', 'starfish', 'wood_rabbit', 'porcupine', 'fox_squirrel', 'beaver', 'guinea_pig', 'zebra',
        'pig', 'hippopotamus', 'bison', 'gazelle', 'llama', 'skunk', 'badger', 'orangutan', 'gorilla', 'chimpanzee',
        'gibbon', 'baboon', 'panda', 'eel', 'clown_fish', 'puffer_fish', 'accordion', 'ambulance', 'assault_rifle',
        'backpack', 'barn', 'wheelbarrow', 'basketball', 'bathtub', 'lighthouse', 'beer_glass', 'binoculars',
        'birdhouse', 'bow_tie', 'broom', 'bucket', 'cauldron', 'candle', 'cannon', 'canoe', 'carousel', 'castle',
        'mobile_phone', 'cowboy_hat', 'electric_guitar', 'fire_engine', 'flute', 'gasmask', 'grand_piano', 'guillotine',
        'hammer', 'harmonica', 'harp', 'hatchet', 'jeep', 'joystick', 'lab_coat', 'lawn_mower', 'lipstick', 'mailbox',
        'missile', 'mitten', 'parachute', 'pickup_truck', 'pirate_ship', 'revolver', 'rugby_ball', 'sandal', 'saxophone',
        'school_bus', 'schooner', 'shield', 'soccer_ball', 'space_shuttle', 'spider_web', 'steam_locomotive', 'scarf',
        'submarine', 'tank', 'tennis_ball', 'tractor', 'trombone', 'vase', 'violin', 'military_aircraft', 'wine_bottle',
        'ice_cream', 'bagel', 'pretzel', 'cheeseburger', 'hotdog', 'cabbage', 'broccoli', 'cucumber', 'bell_pepper',
        'mushroom', 'Granny_Smith', 'strawberry', 'lemon', 'pineapple', 'banana', 'pomegranate', 'pizza', 'burrito',
        'espresso', 'volcano', 'baseball_player', 'scuba_diver', 'acorn'
    ]
    return class_names


def _KD_loss(pred, soft, T):
    pred = torch.log_softmax(pred / T, dim=1)
    soft = torch.softmax(soft / T, dim=1)
    return -1 * torch.mul(soft, pred).sum() / pred.shape[0]


class _LRScheduler(object):
    def __init__(self, optimizer, last_epoch=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.
        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class CosineSchedule(_LRScheduler):

    def __init__(self, optimizer, K):
        self.K = K
        super().__init__(optimizer, -1)

    def cosine(self, base_lr):
        return base_lr * math.cos((99 * math.pi * (self.last_epoch)) / (200 * (self.K - 1)))

    def get_lr(self):
        return [self.cosine(base_lr) for base_lr in self.base_lrs]

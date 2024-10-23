from __future__ import print_function

import math
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import ImageFilter
from statistics import mean,stdev 


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def D(p, z, version='simplified'):  # negative cosine similarity
    if version == 'original':
        z = z.detach()  # stop gradient
        p = F.normalize(p, dim=1)  # l2-normalize
        z = F.normalize(z, dim=1)  # l2-normalize
        return -(p * z).sum(dim=1).mean()

    elif version == 'simplified':  # same thing, much faster. Scroll down, speed test in __main__
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    # import pdb; pdb.set_trace()
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    # print(sim_weight.shape, sim_labels.shape)
    sim_weight = torch.ones_like(sim_weight)

    sim_weight = sim_weight / sim_weight.sum(dim=-1, keepdim=True)

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    # print(one_hot_label.shape)
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    # print(pred_scores.shape)
    pred_labels = pred_scores.argmax(dim=-1)
    return pred_scores, pred_labels

def ball_predict(id,epoch, feature, feature_bank, feature_labels, classes, radius, rule="type1",conf=None,knnweight=False, radaptive=None, otsu_split=None,teto=200):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]

    # import pdb; pdb.set_trace()
    # n1 = torch.norm(feature,dim=1)   # 5000 [B]
    # n2 = torch.norm(feature_bank.T,dim=1)  #50000
    # sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)

    mask = sim_matrix>radius
    sim_indices = torch.nonzero(mask)

    if radaptive == None:
        pass
    elif radaptive=="high":
        temp_radius = 0.99

        while True:
            mask = sim_matrix>temp_radius
            sim_indices = torch.nonzero(mask)
            #if len(sim_indices)<5:
            if len(sim_indices)<40:
            # if len(sim_indices)<20:
                temp_radius -=0.01
            else:
                break

    elif radaptive=="low":
        #temp_radius = 0.75
        temp_radius = 0.3

        while True:
            mask = sim_matrix>temp_radius
            sim_indices = torch.nonzero(mask)
            if len(sim_indices)>200:
                temp_radius +=0.01
            else:
                break
    
    elif ("otsu" in radaptive) and (otsu_split is not None):
        # import pdb; pdb.set_trace()
        if radaptive == "otsu_linear":  #faz com o type_free
            if id in otsu_split['clean_ids']:    
                knn_k = 5
            elif id in otsu_split['maybe_clean_ids']:
                knn_k = 20
            elif id in otsu_split['maybe_noisy_ids']:
                knn_k = 100
            elif id in otsu_split['noisy_ids']:
                knn_k = 200
            else:
                raise Exception("Invalid id")
        elif radaptive == "otsu_linear2":  #faz com o type_free
            if id in otsu_split['clean_ids']:    
                knn_k = 5
            elif id in otsu_split['maybe_clean_ids']:
                knn_k = 20
            elif id in otsu_split['maybe_noisy_ids']:
                knn_k = 50
            elif id in otsu_split['noisy_ids']:
                knn_k = 70
            else:
                raise Exception("Invalid id")
        elif radaptive == "otsu_linear3":  #faz com o type_free
            if id in otsu_split['clean_ids']:    
                if conf < 0.5:
                    knn_k = 40
                else:
                    knn_k = 5
            elif id in otsu_split['maybe_clean_ids']:
                if conf < 0.5:
                    knn_k = 40
                else:
                    knn_k = 20
            elif id in otsu_split['maybe_noisy_ids']:
                knn_k = 50
            elif id in otsu_split['noisy_ids']:
                knn_k = 70
            else:
                raise Exception("Invalid id")
        elif radaptive == "otsu_rad":   #faz com o type_2
            if id in otsu_split['clean_ids']:   
                temp_radius = 0.75 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)>200:
                        temp_radius +=0.01
                    else:
                        break
                
            elif id in otsu_split['maybe_clean_ids']:
                temp_radius = 0.85 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)>200:
                        temp_radius +=0.01
                    else:
                        break
            elif id in otsu_split['maybe_noisy_ids']:
                temp_radius = 0.90 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<5:
                        temp_radius -=0.01
                    else:
                        break
            elif id in otsu_split['noisy_ids']:
                temp_radius = 0.95 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<5:
                        temp_radius -=0.01
                    else:
                        break
            else:
                raise Exception("Invalid id")
            
        elif radaptive == "otsu_rad_inv":   #faz com o type_2
            if id in otsu_split['noisy_ids']:   
                temp_radius = 0.75 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)>200:
                        temp_radius +=0.01
                    else:
                        break
                
            elif id in otsu_split['maybe_noisy_ids']:
                temp_radius = 0.85 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)>200:
                        temp_radius +=0.01
                    else:
                        break
            elif id in otsu_split['maybe_clean_ids']:
                temp_radius = 0.90 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<5:
                        temp_radius -=0.01
                    else:
                        break
            elif id in otsu_split['clean_ids']:
                temp_radius = 0.95 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<5:
                        temp_radius -=0.01
                    else:
                        break
            else:
                raise Exception("Invalid id")

        elif radaptive == "otsu_rad2":   #faz com o type_2
            if id in otsu_split['clean_ids']:   
                temp_radius = 0.99 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<5:
                        temp_radius -=0.01
                    else:
                        break
                
            elif id in otsu_split['maybe_clean_ids']:
                temp_radius = 0.99 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<5:
                        temp_radius -=0.01
                    else:
                        break
            elif id in otsu_split['maybe_noisy_ids'] :
                temp_radius = 0.99
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<40:
                        temp_radius -=0.01
                    else:
                        break
            elif id in otsu_split['noisy_ids']:
                temp_radius = 0.99 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<40:
                        temp_radius -=0.01
                    else:
                        break
            else:
                raise Exception("Invalid id")
            
        elif radaptive == "otsu_rad3":   #faz com o type_2
            if id in otsu_split['clean_ids']:   
                temp_radius = 0.99 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<5:
                        temp_radius -=0.01
                    else:
                        break
                
            elif id in otsu_split['maybe_clean_ids']:
                temp_radius = 0.99 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<20:
                        temp_radius -=0.01
                    else:
                        break
            elif id in otsu_split['maybe_noisy_ids'] :
                temp_radius = 0.99
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<40:
                        temp_radius -=0.01
                    else:
                        break
            elif id in otsu_split['noisy_ids']:
                temp_radius = 0.99 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<80:
                        temp_radius -=0.01
                    else:
                        break
            else:
                raise Exception("Invalid id")
            
        elif radaptive == "otsu_rad4":   #faz com o type_2
            if id in otsu_split['clean_ids']:   
                temp_radius = 0.99 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<5:
                        temp_radius -=0.01
                    else:
                        break
                
            elif id in otsu_split['maybe_clean_ids']:
                temp_radius = 0.99 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<20:
                        temp_radius -=0.01
                    else:
                        break
            elif id in otsu_split['maybe_noisy_ids'] :
                temp_radius = 0.99
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<40:
                        temp_radius -=0.01
                    else:
                        break
            elif id in otsu_split['noisy_ids']:
                temp_radius = 0.99 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<40:
                        temp_radius -=0.01
                    else:
                        break
            else:
                raise Exception("Invalid id")
            
        elif radaptive == "otsu_rad5":   #faz com o type_2
            
            if id in otsu_split['clean_ids']:   
                temp_radius = 0.99 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<5:
                        temp_radius -=0.01
                    else:
                        break
                
            elif id in otsu_split['maybe_clean_ids']:
                temp_radius = 0.99 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<20:
                        temp_radius -=0.01
                    else:
                        break
            elif id in otsu_split['maybe_noisy_ids'] :
                temp_radius = 0.99
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<100:
                        temp_radius -=0.01
                    else:
                        break
            elif id in otsu_split['noisy_ids']:
                temp_radius = 0.99 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<200:
                        temp_radius -=0.01
                    else:
                        break
            else:
                raise Exception("Invalid id")
            
        elif radaptive == "otsu_rad6":   #faz com o type_2
            
            if id in otsu_split['clean_ids']:   
                temp_radius = 0.99 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<20:
                        temp_radius -=0.01
                    else:
                        break
                
            elif id in otsu_split['maybe_clean_ids']:
                temp_radius = 0.99 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<40:
                        temp_radius -=0.01
                    else:
                        break
            elif id in otsu_split['maybe_noisy_ids'] :
                temp_radius = 0.99
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<100:
                        temp_radius -=0.01
                    else:
                        break
            elif id in otsu_split['noisy_ids']:
                temp_radius = 0.99 
                while True:
                    mask = sim_matrix>temp_radius
                    sim_indices = torch.nonzero(mask)
                    if len(sim_indices)<200:
                        temp_radius -=0.01
                    else:
                        break
            else:
                raise Exception("Invalid id")
        
    # knn_k = len(sim_indices)
    # knn_k = max(min,)
    sim_label_topk = False 

    if rule == "type1": 
        # A lógica do tipo 1 é se for menor do que 50, virar knn-50, se for entre 100 e 200, variar,
        # e se for acima disso, limitar pra 200

        #resultados: com 0.95, depois da primeira época, sem tem dado min=50, quer dizer que 0.95 é 
        # muito alto, preciso baixar. Na prática, com knn-50 funciona pior que baseline. A questão de 
        # ficar entre 100 e 200 acabou não sendo explorada.
        
        
        knn_k = min(200,len(sim_indices) )
        if knn_k < 200:

            if knn_k<=50:
                knn_k = 50
                sim_weight, sim_indices = sim_matrix.topk(k=50, dim=-1)
                sim_label_topk = True
                
            else:
                # sim_weight = sim_matrix[mask]
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                sim_label_topk = True
        
        else: # = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

    elif rule == "typefree":
        #free =)
        
        # knn_k = len(sim_indices)
        # sim_weight = sim_matrix[mask]
        # knn_k = min(200,len(sim_indices) )
        
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        sim_label_topk = True

            
    elif rule == "type2":
        #O tipo2 = tamanho livre
        # Resultado 0.95: O mínimo sempre deu 1 (o que eu acho que prejudicou), e o máximo normalmente dá menor que 100
        # Talvez baixando o raio ele melhore.
        knn_k = len(sim_indices)
        # sim_weight = sim_matrix[mask]
        
        
        #if knn_k > 100:
        #if knn_k > 200:
        #teto = 200
        if knn_k > teto:
            # knn_k = 100
            #knn_k = 200
            knn_k = teto
        elif knn_k <5:
            knn_k = 5
        
        sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
        sim_label_topk = True
    elif rule == "type3":
        # O tipo 3 é o que bota o piso pra 50 e o teto pra 200
        # Resultado 0.95: tirando a primeira época, o restou ficou sempre em 50, e virou knn-50
        # Acho que baixando o raio deve melhorar
        knn_k = min(200,len(sim_indices) )
        
        if knn_k<=50:
            knn_k = 50
            sim_weight, sim_indices = sim_matrix.topk(k=50, dim=-1)
            sim_label_topk = True
        else:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True
    elif rule == "type4":
        # A ideia do tipo 4 é botar o piso em 100 e teto em 200
        # Resultado 0.95: Acabou virando knn-100. Acho que diminuindo o valo do raio deve melhorar
        
        knn_k = min(200,len(sim_indices) )
        
        if knn_k<=100:
            knn_k = 100
            sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            sim_label_topk = True
        else:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True
    elif rule == "type5":
        # A ideia desse é ter piso de 50 e teto infinito
        # Resultado 0.95: Acabou virando knn-50 o tempo todo. Diminuir o raio deve melhorar
        knn_k = min(200,len(sim_indices) )
        
        if knn_k<=50:
            knn_k = 50
            sim_weight, sim_indices = sim_matrix.topk(k=50, dim=-1)
            sim_label_topk = True
        else:
            #teste
            knn_k = len(sim_indices)
            # sim_weight = sim_matrix[mask]
            # print(knn_k)
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True

    elif rule == "type6":  #original
        # Deveria dar resultado igual ao original
        # Resultado: sanity check ok
        knn_k = min(200,len(sim_indices) )
        
        
        knn_k = 200
        sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
        sim_label_topk = True

    elif rule == "type4_v2":
        # A ideia do tipo 4 é botar o piso em 100 e teto em 200
        # Resultado 0.95: Acabou virando knn-100. Acho que diminuindo o valo do raio deve melhorar
        
        knn_k = min(200,len(sim_indices) )
        
        if knn_k<=100:
            knn_k = 100
            sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            sim_label_topk = True
        else:
            # knn_k = 00
            
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True

    elif rule == "type7":
        # A ideia do tipo 4 é botar o piso em 100 e teto em 400
        # Resultado 0.95: Acabou virando knn-100. Acho que diminuindo o valo do raio deve melhorar
        
        knn_k = min(400,len(sim_indices) )
        
        if knn_k<=100:
            knn_k = 100
            sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            sim_label_topk = True
        else:
           
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True

    elif rule == "type8":
        # A ideia é ser a primeira metade do treino igual ao baseline (knn=200) e a segunda metade
        # igual ao type4 (knn100 se <100, caso contrário knn200)

        if epoch <=150:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True
        else:
            knn_k = min(200,len(sim_indices) )
        
            if knn_k<=100:
                knn_k = 100
                sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
                sim_label_topk = True
            else:
                knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
                sim_label_topk = True

    elif rule == "type8_inv":
        # A ideia é ser o inverso do type8: a segunda metade do treino igual ao baseline (knn=200) e a primeira metade
        # igual ao type4 (knn100 se <100, caso contrário knn200)

        if epoch >150:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True
        else:
            knn_k = min(200,len(sim_indices) )
        
            if knn_k<=100:
                knn_k = 100
                sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
                sim_label_topk = True
            else:
                knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
                sim_label_topk = True

    elif rule == "type9":
        # A ideia é ser a primeira metade do treino igual ao baseline (knn=200) e a segunda metade
        # igual ao type1 (variar entre 50 e 200)

        if epoch <=150:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True
        else:
            knn_k = min(200,len(sim_indices) )
        
            if knn_k<=50:
                knn_k = 50
                sim_weight, sim_indices = sim_matrix.topk(k=50, dim=-1)
                sim_label_topk = True
            else:
                # knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                sim_label_topk = True

    elif rule == "type9_inv":
        # A ideia é ser a primeira metade do treino igual ao baseline (knn=200) e a segunda metade
        # igual ao type1 (variar entre 50 e 200)

        if epoch >150:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True
        else:
            knn_k = min(200,len(sim_indices) )
        
            if knn_k<=50:
                knn_k = 50
                sim_weight, sim_indices = sim_matrix.topk(k=50, dim=-1)
                sim_label_topk = True
            else:
                # knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                sim_label_topk = True

    elif rule == "type10":
        # A ideia é ser a primeira metade do treino igual ao tipo4 (binario entre 100 e 200) e a segunda metade
        # ser binário entre 100 e 400

        if epoch <=150:
            knn_k = min(200,len(sim_indices) )
            if knn_k<=100:
                knn_k = 100
                sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            else:
                knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True
        else:
            knn_k = min(200,len(sim_indices) )
            if knn_k<=100:
                knn_k = 100
                sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            else:
                knn_k = 400
                sim_weight, sim_indices = sim_matrix.topk(k=400, dim=-1)
            sim_label_topk = True

    elif rule == "type10_v2":
        # A ideia é ser a primeira metade do treino igual ao tipo4 (binario entre 100 e 200) e a segunda metade
        # ser variar entre 100 e 400

        # if epoch <=150:
        if epoch <=70:
            knn_k = min(200,len(sim_indices) )
            if knn_k<=100:
                knn_k = 100
                sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            else:
                knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True
        else:
            knn_k = min(400,len(sim_indices) )
            if knn_k<=100:
                knn_k = 100
                sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            else:
                # knn_k = 400
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True

    elif rule == "type10_v3":
        # A ideia é s igual ao type10_v2:  a primeira metade do treino igual ao tipo4 (binario entre 100 e 200) e a segunda metade
        # ser variar entre 100 e 400, porém começando a segunda "parte mais cedo"

        if epoch <=75:
            knn_k = min(200,len(sim_indices) )
            if knn_k<=100:
                knn_k = 100
                sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            else:
                knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True
        else:
            knn_k = min(400,len(sim_indices) )
            if knn_k<=100:
                knn_k = 100
                sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            else:
                # knn_k = 400
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True

    elif rule == "type10_v4":
        # A ideia é s igual ao type10_v2, mas variando na primerira parte: a primeira metade do treino variando entre 100 e 200 e a segunda metade
        # ser variar entre 100 e 400

        if epoch <=150:
            knn_k = min(200,len(sim_indices) )
            if knn_k<=100:
                knn_k = 100
                sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            else:
                # knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True
        else:
            knn_k = min(400,len(sim_indices) )
            if knn_k<=100:
                knn_k = 100
                sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            else:
                # knn_k = 400
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True

    elif rule == "type10_v5":
        # A ideia é s igual ao type10_v4, mas com piso menor (50): a primeira metade do treino variando entre 50 e 200) e a segunda metade
        # ser variar entre 50 e 400

        if epoch <=150:
            knn_k = min(200,len(sim_indices) )
            if knn_k<=50:
                knn_k = 50
                sim_weight, sim_indices = sim_matrix.topk(k=50, dim=-1)
            else:
                # knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True
        else:
            knn_k = min(400,len(sim_indices) )
            if knn_k<=50:
                knn_k = 50
                sim_weight, sim_indices = sim_matrix.topk(k=50, dim=-1)
            else:
                # knn_k = 400
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True

    elif rule == "type10_v6":
        # A ideia é s igual ao type10_v2:  a primeira metade do treino igual ao tipo4 (binario entre 100 e 200) e a segunda metade
        # ser variar entre 100 e 400, porém começando a segunda "parte mais tarde"

        if epoch <=225:
            knn_k = min(200,len(sim_indices) )
            if knn_k<=100:
                knn_k = 100
                sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            else:
                knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True
        else:
            knn_k = min(400,len(sim_indices) )
            if knn_k<=100:
                knn_k = 100
                sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            else:
                # knn_k = 400
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True

    elif rule == "type10_v7":
        # A ideia é ser igual ao type10_v2, porém com o piso mais alto (150) : a primeira metade do treino igual ao t
        # tipo4 (binario entre 150 e 250) e a segunda metade
        # ser variar entre 150 e 400

        if epoch <=150:
            knn_k = min(250,len(sim_indices) )
            if knn_k<=150:
                knn_k = 150
                sim_weight, sim_indices = sim_matrix.topk(k=150, dim=-1)
            else:
                knn_k = 250
                sim_weight, sim_indices = sim_matrix.topk(k=250, dim=-1)
            sim_label_topk = True
        else:
            knn_k = min(400,len(sim_indices) )
            if knn_k<=150:
                knn_k = 150
                sim_weight, sim_indices = sim_matrix.topk(k=150, dim=-1)
            else:
                # knn_k = 400
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True

    elif rule == "type10_v8":
        # A ideia é ser igual ao type10_v2, porém com o escalas de piso (50/100/150/200) : a primeira metade do treino igual ao t
        # tipo4 (binario entre 50 e 200) e a segunda metade
        # ser variar entre 100 e 400

        if epoch <=150:
            knn_k = min(200,len(sim_indices) )
            if knn_k<=50:
                knn_k = 50
                sim_weight, sim_indices = sim_matrix.topk(k=50, dim=-1)
            elif knn_k<=100:
                knn_k = 100
                sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            elif knn_k<=150:
                knn_k = 150
                sim_weight, sim_indices = sim_matrix.topk(k=150, dim=-1)
            else:
                knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True
        else:
            knn_k = min(400,len(sim_indices) )
            if knn_k<=100:
                knn_k = 100
                sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            else:
                # knn_k = 400
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True

    elif rule == "type10_v9":
        # A ideia é ser igual ao type10_v2, porém com o variando entre 50 e 400 na segunda parte : a primeira metade do treino igual ao t
        # tipo4 (binario entre 10 e 200) e a segunda metade
        # ser variar entre 50 e 400

        if epoch <=150:
            knn_k = min(200,len(sim_indices) )
            if knn_k<=100:
                knn_k = 100
                sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            else:
                knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True
        else:
            knn_k = min(400,len(sim_indices) )
            if knn_k<=50:
                knn_k = 50
                sim_weight, sim_indices = sim_matrix.topk(k=50, dim=-1)
            else:
                # knn_k = 400
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True

    elif rule == "type11":
        # O tipo 11 é o que bota o piso pra 20 e o deixa variar entre 20 e 200
        
        knn_k = min(200,len(sim_indices) )
        
        if knn_k<=20:
            knn_k = 20
            sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
            sim_label_topk = True
        else:
            # knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True

    elif rule == "type11_adaptive":
        # O tipo 11 é o que bota o piso pra 20 e o deixa variar entre 20 e 200, amas adapta o valor do limiar
        
        knn_k = min(200,len(sim_indices) )
        
        if knn_k<=20:
            knn_k = 20
            sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
            sim_label_topk = True
        else:
            # knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True

    elif rule == "type11_inv":
        # O tipo 11 é o que bota o piso pra 20 e o deixa variar entre 20 e 200, porém com condições inversas
        
        knn_k = min(200,len(sim_indices) )
        
        if knn_k>=20:
            knn_k = 20
            sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
            sim_label_topk = True
        else:
            # knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True

    elif rule == "type11_v2":
        # O tipo 11_v2 é o que bota o piso pra 5 e o deixa variar entre 5 e 200
        
        knn_k = min(200,len(sim_indices) )
        
        if knn_k<=5:
            knn_k = 5
            sim_weight, sim_indices = sim_matrix.topk(k=5, dim=-1)
            sim_label_topk = True
        else:
            # knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True

    elif rule == "type11_v3":
        # O tipo 11 é o que bota o piso pra 20 e o deixa variar entre 20 e 200
        
        knn_k = min(400,len(sim_indices) )
        
        if knn_k<=20:
            knn_k = 20
            sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
            sim_label_topk = True
        else:
            # knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
            sim_label_topk = True

    elif rule == "type12":
        # O tipo 12 é o que bota o piso binário pra 100/200 e na segunda metade deixa variar entre 20 e 200
        
        knn_k = min(200,len(sim_indices) )
        if epoch <= 150:
            if knn_k<=20:
                knn_k = 20
                sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
                sim_label_topk = True
            elif knn_k<=100:
                knn_k = 100
                sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
                sim_label_topk = True
            elif knn_k<=200:
                knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
                sim_label_topk = True
            else:
                # knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                sim_label_topk = True
        else: 
            if knn_k<=20:
                knn_k = 20
                sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
                sim_label_topk = True
            else:
                # knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                sim_label_topk = True

    elif rule == "type13":
        # O tipo 13 é o que bota knn 200 nas 5 primeiras épocas, e depois vira tipo11 : piso pra 20 e o deixa variar entre 20 e 200
        
        knn_k = min(200,len(sim_indices) )

        if epoch <= 10:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:
        
            if knn_k<=20:
                knn_k = 20
                sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
                sim_label_topk = True
            else:
                # knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                sim_label_topk = True

    elif rule == "type14":
        # O tipo 14 é o que bota knn 200 nas  primeiras épocas, e depois vira knnn 20
        
        # knn_k = min(200,len(sim_indices) )

        if epoch <= 100:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:
        
            knn_k = 20
            sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
            sim_label_topk = True
    elif rule == "type14_v2":
        # O tipo 14 é o que bota knn 200 nas  primeiras épocas, e depois vira knnn 20
        
        # knn_k = min(200,len(sim_indices) )

        if epoch <= 70:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:
        
            knn_k = 20
            sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
            sim_label_topk = True
    elif rule == "type14_v3":
        # O tipo 14 é o que bota knn 200 nas  primeiras épocas, e depois vira knnn 20
        
        # knn_k = min(200,len(sim_indices) )

        if epoch <= 50:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True
        elif epoch <=100:
            knn_k = 100
            sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            sim_label_topk = True

        else:
        
            knn_k = 20
            sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
            sim_label_topk = True
    elif rule == "type14_v4":
        # O tipo 14 é o que bota knn 200 nas  primeiras épocas, e depois vira knnn 20
        
        # knn_k = min(200,len(sim_indices) )

        if epoch <= 70:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:
        
            knn_k = 100
            sim_weight, sim_indices = sim_matrix.topk(k=100, dim=-1)
            sim_label_topk = True
    elif rule == "type14_inv":
        # O tipo 14 é o que bota knn 200 nas  primeiras épocas, e depois vira knnn 20
        
        # knn_k = min(200,len(sim_indices) )

        if epoch >= 100:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:
        
            knn_k = 20
            sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
            sim_label_topk = True
    elif rule == "type15_v1":
        # O tipo 15 é baseado na confiança das predições: conf<0.9: knn 200, senao knn 20
        

        if conf <0.9:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:
        
            knn_k = 20
            sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
            sim_label_topk = True

    elif rule == "type15_v2":
        # O tipo 15 é baseado na confiança das predições: conf<0.8: knn 200, senao knn 20
        

        if conf <0.8:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:
        
            knn_k = 20
            sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
            sim_label_topk = True

    elif rule == "type15_v3":
        # O tipo 15 é baseado na confiança das predições: conf<0.95: knn 200, senao knn 20
        

        if conf <0.95:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:
        
            knn_k = 20
            sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
            sim_label_topk = True


    elif rule == "type15_v5":
        # O tipo 15 é baseado na confiança das predições: conf<0.9: knn 200, senao knn 20
        knn_k = min(200,len(sim_indices) )

        if conf <0.9:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:
        
            if knn_k<=20:
                knn_k = 20
                sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
                sim_label_topk = True
            else:
                # knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                sim_label_topk = True

    elif rule == "type15_v6":
        # O tipo 15 é baseado na confiança das predições: conf<0.9: knn 200, senao knn 20
        knn_k = min(200,len(sim_indices) )

        if conf <0.95:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:
        
            if knn_k<=20:
                knn_k = 20
                sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
                sim_label_topk = True
            else:
                # knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                sim_label_topk = True

    elif rule == "type15_v7":
        # O tipo 15 é baseado na confiança das predições: conf<0.9: knn 200, senao knn 20
        knn_k = min(200,len(sim_indices) )

        if conf <0.95:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:
        
            if knn_k<=5:
                knn_k = 5
                sim_weight, sim_indices = sim_matrix.topk(k=5, dim=-1)
                sim_label_topk = True
            else:
                # knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                sim_label_topk = True

    elif rule == "type15_v8":
        # O tipo 15 é baseado na confiança das predições: conf<0.9: knn 200, senao knn 20
        knn_k = min(200,len(sim_indices) )

        if conf <0.9:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:
        
            if knn_k<=5:
                knn_k = 5
                sim_weight, sim_indices = sim_matrix.topk(k=5, dim=-1)
                sim_label_topk = True
            else:
                # knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                sim_label_topk = True

    elif rule == "type15_v9":
        # O tipo 15 é baseado na confiança das predições: conf<0.9: knn 200, senao knn 20
        knn_k = min(200,len(sim_indices) )

        if conf <0.8:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:
        
            if knn_k<=5:
                knn_k = 5
                sim_weight, sim_indices = sim_matrix.topk(k=5, dim=-1)
                sim_label_topk = True
            else:
                # knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                sim_label_topk = True

    elif rule == "type15_v10":
        # O tipo 15 é baseado na confiança das predições: conf<0.9: knn 200, senao knn 20
        knn_k = min(200,len(sim_indices) )

        if conf <0.8:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:
        
            if knn_k<=20:
                knn_k = 20
                sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
                sim_label_topk = True
            else:
                # knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                sim_label_topk = True

    elif rule == "type15_v11":
        # O tipo 15 é baseado na confiança das predições: conf<0.9: knn 200, senao knn 20
        knn_k = min(200,len(sim_indices) )

        if conf <0.7:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:
        
            if knn_k<=5:
                knn_k = 5
                sim_weight, sim_indices = sim_matrix.topk(k=5, dim=-1)
                sim_label_topk = True
            else:
                # knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                sim_label_topk = True

    elif rule == "type15_v12":
        # O tipo 15 é baseado na confiança das predições: conf<0.9: knn 200, senao knn 20
        knn_k = min(200,len(sim_indices) )

        if conf <0.6:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:
        
            if knn_k<=5:
                knn_k = 5
                sim_weight, sim_indices = sim_matrix.topk(k=5, dim=-1)
                sim_label_topk = True
            else:
                # knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                sim_label_topk = True

    elif rule == "type15_v13":
        # O tipo 15 é baseado na confiança das predições: conf<0.9: knn 200, senao knn 20
        knn_k = min(200,len(sim_indices) )

        if conf <0.7:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:
        
            if knn_k<=20:
                knn_k = 20
                sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
                sim_label_topk = True
            else:
                # knn_k = 200
                sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                sim_label_topk = True

    elif rule == "type15_v14":
        # O tipo 15 é baseado na confiança das predições: conf<0.9: knn 200, senao knn 20
        knn_k = min(200,len(sim_indices) )

        if conf <0.7:
            knn_k = 200
            sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)
            sim_label_topk = True

        else:

            if conf >=0.9:
                knn_k = 5
                sim_weight, sim_indices = sim_matrix.topk(k=5, dim=-1)
                sim_label_topk = True
            elif conf >=0.8:
        
                if knn_k<=5:
                    knn_k = 5
                    sim_weight, sim_indices = sim_matrix.topk(k=5, dim=-1)
                    sim_label_topk = True
                else:
                    # knn_k = 200
                    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                    sim_label_topk = True
            else: 
                if knn_k<=20:
                    knn_k = 20
                    sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
                    sim_label_topk = True
                else:
                    # knn_k = 200
                    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                    sim_label_topk = True

    elif rule == "type15_v15":
        # O tipo 15 é baseado na confiança das predições: conf<0.9: knn 200, senao knn 20
        knn_k = min(200,len(sim_indices) )

        if conf <0.5:
            knn_k = 40
            sim_weight, sim_indices = sim_matrix.topk(k=40, dim=-1)
            sim_label_topk = True

        else:

            if conf >=0.8:
                knn_k = 5
                sim_weight, sim_indices = sim_matrix.topk(k=5, dim=-1)
                sim_label_topk = True
            elif conf >=0.7:
        
                if knn_k<=5:
                    knn_k = 5
                    sim_weight, sim_indices = sim_matrix.topk(k=5, dim=-1)
                    sim_label_topk = True
                else:
                    # knn_k = 200
                    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                    sim_label_topk = True
            else: 
                if knn_k<=20:
                    knn_k = 20
                    sim_weight, sim_indices = sim_matrix.topk(k=20, dim=-1)
                    sim_label_topk = True
                else:
                    # knn_k = 200
                    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
                    sim_label_topk = True

    

    

    


    
        

    #sim_weight=sim_matrix[sim_indices]
    # print(knn_k)
    # import pdb; pdb.set_trace()
    

    

    
    # [B, K]
    #sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    if sim_label_topk == True:
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    else:
        sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices[:,1].view(1,-1))
    # print(sim_weight.shape, sim_labels.shape)
    
    if knnweight:
        #using real weights based on limilarity [frc]
        sim_weight = (sim_weight)/sim_weight.sum(-1)

    else:

        sim_weight = torch.ones_like(sim_weight)

        sim_weight = sim_weight / sim_weight.sum(dim=-1, keepdim=True)
    

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    # print(one_hot_label.shape)
    # import pdb; pdb.set_trace()
    
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
    # print(pred_scores.shape)
    pred_labels = pred_scores.argmax(dim=-1)
    return pred_scores, pred_labels, knn_k

def aknn_predict(id, feature, feature_bank, feature_labels, classes, rule="type1", radaptive=None, otsu_split=None,teto=200, kmin1=40, kmin2=80):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # sim_matrix = torch.mm(feature, feature_bank.T)  # Transpose feature_bank for matrix multiplication
    # [B, K]

    # n1 = torch.norm(feature,dim=1)   # 5000 [B]
    # n2 = torch.norm(feature_bank.T,dim=1)  #50000
    # sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # sim_weight, sim_indices = sim_matrix.topk(k=200, dim=-1)

    # mask = sim_matrix>radius
    # sim_indices = torch.nonzero(mask)

    
    # import pdb; pdb.set_trace()
    # pred_score = torch.tensor([]).to(feature.device)
    # pred_labels = torch.tensor([], dtype=torch.long).to(feature.device)

    pred_score = torch.zeros((feature.size(0), classes), device=feature.device)
    pred_labels = torch.zeros((feature.size(0),), dtype=torch.long, device=feature.device)

    # temp_radius = 0.99 
    # mask = sim_matrix>temp_radius
    # kmask = torch.sum(sim_matrix>temp_radius,-1)
    #step = 0.01
    step = 0.03
    
    for id in range(sim_matrix.size(0)):
        # if ("otsu" in radaptive) and (otsu_split is not None):
        # temp_radius = 0.99 
    
        # if radaptive == "otsu_rad3":   #faz com o type_2
        if id in otsu_split['clean_ids']:   
            temp_radius = 0.99 
            
            while True:
                mask = sim_matrix[id]>temp_radius
                sim_indices = torch.nonzero(mask)
                if len(sim_indices)<5:
                    #temp_radius -=0.01
                    temp_radius -=step
                else:
                    break
            
            # while len(torch.nonzero(mask[id]))<5:
            #     temp_radius -=0.01
            #     mask[id] = sim_matrix[id]>temp_radius


            
        elif id in otsu_split['maybe_clean_ids']:
            temp_radius = 0.99 
            while True:
                mask = sim_matrix[id]>temp_radius
                sim_indices = torch.nonzero(mask)
                if len(sim_indices)<20:
                    #temp_radius -=0.01
                    temp_radius -=step
                else:
                    break
            # while len(torch.nonzero(mask[id]))<20:
            #     temp_radius -=0.01
            #     mask[id] = sim_matrix[id]>temp_radius

        elif id in otsu_split['maybe_noisy_ids'] :
            temp_radius = 0.99
            while True:
                mask = sim_matrix[id]>temp_radius
                sim_indices = torch.nonzero(mask)
                #if len(sim_indices)<40:
                if len(sim_indices)<kmin1:
                    #temp_radius -=0.01
                    temp_radius -=step
                else:
                    break
            # while len(torch.nonzero(mask[id]))<40:
            #     temp_radius -=0.01
            #     mask[id] = sim_matrix[id]>temp_radius
        elif id in otsu_split['noisy_ids']:
            temp_radius = 0.99 
            while True:
                mask = sim_matrix[id]>temp_radius
                sim_indices = torch.nonzero(mask)
                #if len(sim_indices)<80:
                if len(sim_indices)<kmin2:
                    #temp_radius -=0.01
                    temp_radius -=step
                else:
                    break
            # while len(torch.nonzero(mask[id]))<80:
            #     temp_radius -=0.01
            #     mask[id] = sim_matrix[id]>temp_radius
        else:
            raise Exception("Invalid id")
                
            
        # sim_indices = torch.nonzero(mask[id])
        sim_label_topk = False 

                
        if rule == "type2":
            #O tipo2 = tamanho livre
            # Resultado 0.95: O mínimo sempre deu 1 (o que eu acho que prejudicou), e o máximo normalmente dá menor que 100
            # Talvez baixando o raio ele melhore.
            knn_k = len(sim_indices)
            # sim_weight = sim_matrix[mask]
            
            #teto = 200
            if knn_k > teto:            
                knn_k = teto
            elif knn_k <5:
                knn_k = 5
            
            sim_weight, sim_indices = sim_matrix[id].topk(k=knn_k, dim=-1)
            sim_label_topk = True
        
            
        #sim_weight=sim_matrix[sim_indices]
        # print(knn_k)
        
        
        # [B, K]
        #sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)


        if sim_label_topk == True:
            sim_labels = torch.gather(feature_labels.expand(1, -1), dim=-1, index=sim_indices.unsqueeze(0))
        
        
        # else:
        #     sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices[:,1].view(1,-1))
        # print(sim_weight.shape, sim_labels.shape)
        # import pdb; pdb.set_trace()

        sim_weight = torch.ones_like(sim_weight)

        # sim_weight = sim_weight / sim_weight.sum(dim=-1, keepdim=True)
        sim_weight /= sim_weight.sum(dim=-1, keepdim=True)
        

        # counts for each class
        #one_hot_label = torch.zeros(feature[id].size(0) * knn_k, classes, device=sim_labels.device)

        one_hot_label = torch.zeros(1 * knn_k, classes, device=sim_labels.device)
        # one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)

        one_hot_label.scatter_(dim=-1, index=sim_labels.view(-1, 1), value=1.0)

        # weighted score ---> [B, C]
        # print(one_hot_label.shape)
        
        #pred_scores = torch.sum(one_hot_label.view(feature[id].size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
        #part_score = torch.sum(one_hot_label.view(feature[id].size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
        part_score = torch.sum(one_hot_label.view(1, -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)
        # print(pred_scores.shape)
        #pred_labels = pred_scores.argmax(dim=-1)
        part_pred = part_score.argmax(dim=-1)

        # pred_score = torch.cat([pred_score, part_score], dim=0)
        # pred_labels = torch.cat([pred_labels, part_pred], dim=0)
        pred_score[id] = part_score
        pred_labels[id] = part_pred

    #return pred_score, pred_labels, knn_k
    return pred_score, pred_labels

def fast_weighted_knn_ball(epoch, cur_feature, feature, label, num_classes, chunks=10, norm='global', rule="type1",conf=None, radaptive=None, otsu_split=None,teto=200, kmin1=40, kmin2=80):
    # distributed fast KNN and sample selection with three different modes
    num = len(cur_feature)
    num_class = torch.tensor([torch.sum(label == i).item() for i in range(num_classes)]).to(
        feature.device) + 1e-10
    pi = num_class / num_class.sum()
    split = torch.tensor(np.linspace(0, num, chunks + 1, dtype=int), dtype=torch.long).to(feature.device)
    
    # it will be a sample at a time because the neighborhood change be differt for each sample
    # split = torch.tensor(np.linspace(0, num, num+1, dtype=int), dtype=torch.long).to(feature.device)
    score = torch.tensor([]).to(feature.device)
    pred = torch.tensor([], dtype=torch.long).to(feature.device)
    feature = torch.nn.functional.normalize(feature, dim=1)
    # knn_min = 100000
    # knn_max = -1
    # knn_hist = []
    with torch.no_grad():
        for i in range(chunks):
        #for i in range(num-1):
        # for i in range(num):
            torch.cuda.empty_cache()
            part_feature = cur_feature[split[i]: split[i + 1]]

            # if type == "normal":
            #     part_score, part_pred = knn_predict(part_feature, feature.T, label, num_classes, knn_k)
            # elif type == "ball":
                # knn_k is the radius
            # if conf is None:
            #     part_score, part_pred, k_value = aknn_predict(i,epoch, part_feature, feature.T, label, num_classes, rule,radaptive=radaptive,teto=teto)
            # else:
            #part_score, part_pred, k_value = aknn_predict(i, part_feature, feature.T, label, num_classes, rule,radaptive=radaptive,otsu_split=otsu_split,teto=teto)
            #part_score, part_pred = aknn_predict(i, part_feature, feature.T, label, num_classes, rule,radaptive=radaptive,otsu_split=otsu_split,teto=teto)
            part_score, part_pred = aknn_predict(i, part_feature, feature.T, label, num_classes, rule,radaptive=radaptive,otsu_split=otsu_split,teto=teto, kmin1=kmin1, kmin2=kmin2)
            # part_score, part_pred = knn_predict(part_feature, feature.T, label, num_classes, 200)
            # k_value=200

            score = torch.cat([score, part_score], dim=0)
            pred = torch.cat([pred, part_pred], dim=0)
            # knn_min = min(k_value, knn_min)
            # knn_max = max(k_value, knn_max)
            # knn_hist.append(k_value)
            
        # knn_mean = mean(knn_hist)
        # knn_std = stdev(knn_hist)
        # balanced vote
        if norm == 'global':
            # global normalization
            score = score / pi
        else:  # no normalization
            pass
        score = score/score.sum(1, keepdim=True)

        #pred_scores2 = pred_scores/pred_scores.sum(1, keepdim=True)
    # print(f'knn_min: {knn_min} knn_max: {knn_max}')

    #return score, knn_min, knn_max, knn_mean, knn_std   # , pred
    return score # , pred

def weighted_knn_ball(epoch, cur_feature, feature, label, num_classes, knn_k=100, chunks=10, norm='global', radius=0.99, rule="type1",conf=None, knnweight=False, radaptive=None, otsu_split=None,teto=200):
    # distributed fast KNN and sample selection with three different modes
    num = len(cur_feature)
    num_class = torch.tensor([torch.sum(label == i).item() for i in range(num_classes)]).to(
        feature.device) + 1e-10
    pi = num_class / num_class.sum()
    #split = torch.tensor(np.linspace(0, num, chunks + 1, dtype=int), dtype=torch.long).to(feature.device)
    
    # it will be a sample at a time because the neighborhood change be differt for each sample
    split = torch.tensor(np.linspace(0, num, num+1, dtype=int), dtype=torch.long).to(feature.device)
    score = torch.tensor([]).to(feature.device)
    pred = torch.tensor([], dtype=torch.long).to(feature.device)
    feature = torch.nn.functional.normalize(feature, dim=1)
    knn_min = 100000
    knn_max = -1
    knn_hist = []
    with torch.no_grad():
        #for i in range(chunks):
        #for i in range(num-1):
        for i in range(num):
            torch.cuda.empty_cache()
            part_feature = cur_feature[split[i]: split[i + 1]]

            # if type == "normal":
            #     part_score, part_pred = knn_predict(part_feature, feature.T, label, num_classes, knn_k)
            # elif type == "ball":
                # knn_k is the radius
            if conf is None:
                part_score, part_pred, k_value = ball_predict(i,epoch, part_feature, feature.T, label, num_classes, radius, rule,knnweight=knnweight,radaptive=radaptive,teto=teto)
            else:
                part_score, part_pred, k_value = ball_predict(i,epoch, part_feature, feature.T, label, num_classes, radius, rule,conf[i],knnweight=knnweight,radaptive=radaptive,otsu_split=otsu_split,teto=teto)
            score = torch.cat([score, part_score], dim=0)
            pred = torch.cat([pred, part_pred], dim=0)
            knn_min = min(k_value, knn_min)
            knn_max = max(k_value, knn_max)
            knn_hist.append(k_value)
            
        knn_mean = mean(knn_hist)
        knn_std = stdev(knn_hist)
        # balanced vote
        if norm == 'global':
            # global normalization
            score = score / pi
        else:  # no normalization
            pass
        score = score/score.sum(1, keepdim=True)

        #pred_scores2 = pred_scores/pred_scores.sum(1, keepdim=True)
    print(f'knn_min: {knn_min} knn_max: {knn_max}')

    return score, knn_min, knn_max, knn_mean, knn_std   # , pred

def weighted_knn(cur_feature, feature, label, num_classes, knn_k=100, chunks=10, norm='global', type='knn'):
    # distributed fast KNN and sample selection with three different modes
    num = len(cur_feature)
    num_class = torch.tensor([torch.sum(label == i).item() for i in range(num_classes)]).to(
        feature.device) + 1e-10
    pi = num_class / num_class.sum()
    split = torch.tensor(np.linspace(0, num, chunks + 1, dtype=int), dtype=torch.long).to(feature.device)
    score = torch.tensor([]).to(feature.device)
    pred = torch.tensor([], dtype=torch.long).to(feature.device)
    feature = torch.nn.functional.normalize(feature, dim=1)
    with torch.no_grad():
        for i in range(chunks):
            # import pdb; pdb.set_trace()
            torch.cuda.empty_cache()
            
            part_feature = cur_feature[split[i]: split[i + 1]]

            if type == "knn":
                part_score, part_pred = knn_predict(part_feature, feature.T, label, num_classes, knn_k)
                
            elif type == "ball":
                # knn_k is the radius
                part_score, part_pred = ball_predict(part_feature, feature.T, label, num_classes, knn_k)
            score = torch.cat([score, part_score], dim=0)
            pred = torch.cat([pred, part_pred], dim=0)

        # balanced vote
        if norm == 'global':
            # global normalization
            score = score / pi
        else:  # no normalization
            pass
        score = score/score.sum(1, keepdim=True)

    return score  # , pred

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_config(args, path):
    dict = vars(args)
    if not os.path.isdir(path):
        os.mkdir(os.path.abspath(path))
    with open(path + '/params.csv', 'w') as f:
        for key in dict.keys():
            f.write("%s\t%s\n" % (key, dict[key]))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class KCropsTransform:
    """Take K random crops of one image as the query and key."""

    def __init__(self, base_transform, K=2):
        self.base_transform = base_transform
        self.K = K

    def __call__(self, x):
        res = [self.base_transform(x) for i in range(self.K)]
        return res


class MixTransform:
    def __init__(self, strong_transform, weak_transform, K=2):
        self.strong_transform = strong_transform
        self.weak_transform = weak_transform
        self.K = K

    def __call__(self, x):
        res = [self.weak_transform(x) for i in range(self.K)] + [self.strong_transform(x) for i in range(self.K)]
        return res


class DMixTransform:
    def __init__(self, transforms, nums):
        self.transforms = transforms
        self.nums = nums

    def __call__(self, x):
        res = []
        for i, trans in enumerate(self.transforms):
            res += [trans(x) for _ in range(self.nums[i])]
        return res


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
def otsu(p):
    """ Calculates a threshold to separate into two groups based on OTSU """
    hist = torch.histc(p, 1000, 0, 1)

    def Q(hist, s, mu):
        n1, n2 = hist[:s].sum(), hist[s:].sum()
        if n1 == 0 or n2 == 0:
            return -10000000, None, None, None, None
        mu1 = (hist[0:s] * (torch.arange(0, s).to(hist.device) / 1000)).sum() / n1
        mu2 = (hist[s:1000] * (torch.arange(s, 1000).to(hist.device) / 1000)).sum() / n2
        sigma1 = ((hist[0:s] * ((torch.arange(0, s).to(hist.device) / 1000) - mu1).pow(2)).sum() / n1).clamp(
            0.00001).sqrt()
        sigma2 = ((hist[s:1000] * ((torch.arange(s, 1000).to(hist.device) / 1000) - mu2).pow(2)).sum() / n2).clamp(
            0.00001).sqrt()
        q = (n1 * (mu1 - mu).pow(2) + n2 * (mu2 - mu).pow(2)) / (n1 * sigma1 + n2 * sigma2)
        return q, mu1, mu2, sigma1, sigma2

    q = [0, 0]
    for s in range(2, 998):
        q.append(Q(hist, s, p.mean())[0])
    s = torch.argmax(torch.tensor(q))
    q, mu1, mu2, sigma1, sigma2 = Q(hist, s, p.mean())
    # import pdb; pdb.set_trace()
    mu2, sigma2, mu1, sigma1, s = mu2.detach().cpu().item(), sigma2.detach().cpu().item(), \
                                  mu1.detach().cpu().item(), sigma1.detach().cpu().item(), s / 1000
    return mu2, sigma2, mu1, sigma1, s

def split_dataset(p):
    # Convert logits to probabilities
    up_mean, up_sigma, lo_mean, lo_sigma, thresh = otsu(p)
    print(f'otsu split performed:\n\tup_mean: {up_mean:0.5f}\tup_sigma: {up_sigma:0.5f}\n\tlo_mean: {lo_mean:0.5f}\tlo_sigma: {lo_sigma:0.5f}\n\tthresh:  {thresh:0.5f}\n')
    # Search if we are in overfitting region
    group_4 = p < lo_mean
    group_3 = ~group_4 * (p < thresh)
    group_1 = p >= up_mean
    group_2 = ~group_1 * (p >= thresh)

    return group_1, group_2, group_3, group_4

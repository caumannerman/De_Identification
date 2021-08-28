import torch
import torchvision
from torchvision import models
import torchvision.transforms as T

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

THRESHOLD = 0.95
IMG_SIZE = 480

model = models.detection.keypointrcnn_resnet50_fpn(pretrained=True).eval()

#Add file name
file_loc = ""
## 이미지
image_list = os.listdir(file_loc)
#순서대로 input해주기 위해 sort()
image_list.sort()
trf = T.Compose([
        T.ToTensor()
    ])

dots_5 = np.array([])
dots_5_dist = np.array([])
tick = True
under_shoulder = 0

img = Image.open(file_loc+'/' + image_list[1])
img = img.resize((IMG_SIZE, int(img.height * IMG_SIZE / img.width)))
real_img = np.array(img)
# 이전 타원 이미지 위한 것
#ellip_img_former = np.zeros((real_img.shape[0], real_img.shape[1]))
ellip_img = np.zeros((real_img.shape[0], real_img.shape[1]))

# 맥에서의 .DS_STORE를 제거하기 위해 1부터
for i in range(616,len(image_list)):

    # 원본 image
    print(i)
    img = Image.open(file_loc+'/' + image_list[i])
    img = img.resize((IMG_SIZE, int(img.height * IMG_SIZE / img.width)))
    real_img = np.array(img)

    # 눈,코 개별 blur에 사용하기 위한 이미지 가로, 세로 최대 길이
    real_img_height, real_img_width = real_img.shape[0], real_img.shape[1]
    # 원본을 blur처리한 image
    gauss_img = cv2.GaussianBlur(real_img, (13, 13), 300, 300)
    # 검은 화면에 타원만 그린 image인 ellip_img 갱신은 밑에서 이전 ellip이미지가 더 필요없어지면 그 떄 갱신해줄 것이다. ( 일단 주석처리 )
    #ellip_img = np.zeros((real_img.shape[0], real_img.shape[1]))

    out = model([trf(img)])[0]


    tick = True
    for box, score, keypoints in zip(out['boxes'], out['scores'], out['keypoints']):
        score = score.detach().numpy()

        if score < THRESHOLD:
            continue
        dots_5 = np.array([])
        dots_5_dist = np.array([])
        box = box.detach().numpy()
        keypoints = keypoints.detach().numpy()

        nose_x = int(keypoints[0][0])
        nose_y = int(keypoints[0][1])
        left_ear_x = int(keypoints[3][0])
        right_ear_x = int(keypoints[4][0])
        left_knee_x = int(keypoints[13][0])
        right_knee_x = int(keypoints[14][0])
        left_pelvis_x = int(keypoints[11][0])
        right_pelvis_x = int(keypoints[12][0])
        left_shoulder_x = int(keypoints[5][0])
        left_shoulder_y = int(keypoints[5][1])
        right_shoulder_x = int(keypoints[6][0])
        right_shoulder_y = int(keypoints[6][1])
        left_elbow_x = int(keypoints[7][0])
        right_elbow_x = int(keypoints[8][0])

        # tick을 사용하는 이유 : 한 frame 안에서 가장 확률 높은 박스에 대해서만 눈,코,귀 위치를 적어주기 위함
        if tick:
            cv2.putText(real_img, 'nose=' + str(nose_x) + ',' + str(nose_y), (10, 15), 1, 1, (100, 255, 255), 1,
                        cv2.LINE_AA)
            cv2.putText(real_img, 'leye=' + str(int(keypoints[1][0])) + ',' + str(int(keypoints[1][1])), (10, 29), 1, 1,
                        (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(real_img, 'reye=' + str(int(keypoints[2][0])) + ',' + str(int(keypoints[2][1])), (10, 43), 1, 1,
                        (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(real_img, 'lear=' + str(left_ear_x) + ',' + str(int(keypoints[3][1])), (10, 57), 1, 1,
                        (255,0,0 ), 1, cv2.LINE_AA)
            cv2.putText(real_img, 'rear=' + str(right_ear_x) + ',' + str(int(keypoints[4][1])), (10, 71), 1, 1,
                        ( 255,0,0 ), 1, cv2.LINE_AA)
            tick = False



            # 3 번이 왼쪽 귀, 4 번이 오른쪽 귀, 13번이 왼쪽 무릎,  14번이 오른쪽 무릎
        #########################   여기서, 사진 속 여러 인물 중,  해당 인물이 뒷모습인 경우, 다른 인물 (box)로 넘어가야하기 때문에 continue #########################
        # 뒷모습인 경우만 걸러주는 곳
        if left_knee_x < right_knee_x and left_pelvis_x < right_pelvis_x and left_elbow_x < left_shoulder_x < right_shoulder_x:
            if right_elbow_x < right_shoulder_x and right_elbow_x < right_pelvis_x:
                cv2.circle(real_img, center=tuple(keypoints[0][:2].astype(int)), radius=1, color=(255, 150, 205),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[1][:2].astype(int)), radius=1, color=(205, 100, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[2][:2].astype(int)), radius=1, color=(0, 190, 225),
                           thickness=-1)
                # 왼쪽 귀 - 보라? 핑크?
                cv2.circle(real_img, center=tuple(keypoints[3][:2].astype(int)), radius=1, color=(255, 0, 255),
                           thickness=-1)
                # 오른쪽 귀 - 빨강
                cv2.circle(real_img, center=tuple(keypoints[4][:2].astype(int)), radius=1, color=(0, 0, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[5][:2].astype(int)), radius=1, color=(255, 0, 0),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[11][:2].astype(int)), radius=1, color=(100, 255, 100),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[13][:2].astype(int)), radius=1, color=(255, 255, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[7][:2].astype(int)), radius=1, color=(255,0,0),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[8][:2].astype(int)), radius=1, color=(0,255,0),
                           thickness=-1)
                print("뒷모습1")
                continue
            # 이곳은 왼 팔꿈치가 왼 골반 바깥쪽에, 오른 팔꿈치가 오른 골반 바깥쪽에 존재하는 경우이다. - 고개가 조금 틀어지는 경우 고려해야함(blur)
            # 즉, blur를 하면 안되는 경우이다.
            elif left_elbow_x < left_pelvis_x:
                cv2.circle(real_img, center=tuple(keypoints[0][:2].astype(int)), radius=1, color=(255, 150, 205),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[1][:2].astype(int)), radius=1, color=(205, 100, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[2][:2].astype(int)), radius=1, color=(0, 190, 225),
                           thickness=-1)
                # 왼쪽 귀 - 보라? 핑크?
                cv2.circle(real_img, center=tuple(keypoints[3][:2].astype(int)), radius=1, color=(255, 0, 255),
                           thickness=-1)
                # 오른쪽 귀 - 빨강
                cv2.circle(real_img, center=tuple(keypoints[4][:2].astype(int)), radius=1, color=(0, 0, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[5][:2].astype(int)), radius=1, color=(255, 0, 0),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[11][:2].astype(int)), radius=1, color=(100, 255, 100),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[13][:2].astype(int)), radius=1, color=(255, 255, 255),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[7][:2].astype(int)), radius=1, color=(255, 0, 0),
                           thickness=-1)
                cv2.circle(real_img, center=tuple(keypoints[8][:2].astype(int)), radius=1, color=(0, 255, 0),
                           thickness=-1)
                # 애매하게 뒷모습인 경우 blur를 해야하는데, blur박스의 크기가 본 이미지를 벗어날 수 있기 때문
                for jjj in range(3):
                    now_x, now_y = keypoints[jjj][:2].astype(int)
                    lll, rrr, uuu, ddd = now_x - 3, now_x + 4, now_y - 3, now_y + 4
                    lll = now_x - 3 if now_x > 3 else 0
                    rrr = now_x + 4 if now_x + 4 <= real_img_width else real_img_width
                    uuu = now_y - 3 if now_y > 3 else 0
                    ddd = now_y if now_y + 4 <= real_img_height else real_img_height
                    real_img[uuu: ddd, lll: rrr] = cv2.GaussianBlur(real_img[uuu: ddd, lll: rrr], (9, 9), 30)
                print("뒷모습2")
                continue
            else:
                pass

###########################################   뒷모습 걸러주는 Rule "END"   ################################################


##########################################  얼굴이 제대로 안 잡힌 경우 처리해주기 "START" #######################################
        for ii in range(5):
            dots_5 = np.append(dots_5, np.array([int(keypoints[ii][0]), int(keypoints[ii][1])]))
            #cv2.circle(img, center=tuple(keypoints[j].astype(int)), radius=1, color=(0, 255, 0), thickness=-1)

        dots_5 = dots_5.reshape((5, 2))

        # 얼굴 catch 오류 1 :눈, 코, 귀 중 두 점 이상이 양쪽 어깨보다 낮게 위치하면 이전 이미지의 blur위치 가져옴 ( 모델이 눈,코,귀를 잘못 잡았다고 판단 )
        under_shoulder = 0
        for kkk in dots_5[:,1]:
            if kkk > left_shoulder_y or kkk > right_shoulder_y:
                under_shoulder += 1

        if under_shoulder >= 2:
            # 타원 모자이크 vectorize Complete
            for j in range(real_img.shape[0]):
                ## 이전 타원 정보인 ellip_img를 가져와서 그려줌
                # 다 깜장색임 (타원의 일부가 없음) -> 그냥 real_img 그대로 감
                if not ellip_img[j].any():
                    continue
                else:
                    real_img[j] = [real_img[j][k] if not ellip_img[j][k] else gauss_img[j][k] for k in range(real_img.shape[1])]

            # 타원이 어긋남을 대비한 눈,코 개별 모자이크 - 없앰 why? - 잘못 잡은 점들로 개별 모자이크 할 필요가 없다.

            # 모자이크 위에 눈,코,귀 찍어봄 - 어떻게
            cv2.circle(real_img, center=tuple(keypoints[0][:2].astype(int)), radius=1, color=(255, 150, 205),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[1][:2].astype(int)), radius=1, color=(205, 100, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[2][:2].astype(int)), radius=1, color=( 0, 190,225),
                       thickness=-1)
            #왼쪽 귀 - 보라? 핑크?
            cv2.circle(real_img, center=tuple(keypoints[3][:2].astype(int)), radius=1, color=(255, 0, 255),thickness=-1)
            #오른쪽 귀 - 빨강
            cv2.circle(real_img, center=tuple(keypoints[4][:2].astype(int)), radius=1, color=(0, 0, 255),thickness=-1)

            cv2.circle(real_img, center=tuple(keypoints[5][:2].astype(int)), radius=1, color=(255, 0, 0), thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[11][:2].astype(int)), radius=1, color=(100, 255, 100),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[13][:2].astype(int)), radius=1, color=(255, 255, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[7][:2].astype(int)), radius=1, color=(255, 0, 0),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[8][:2].astype(int)), radius=1, color=(0, 255, 0),
                       thickness=-1)
            break

        # 얼굴 catch 오류 2 : 왼, 오 사이가 너무 길면 눈,코,귀 잘못 잡은 것으로 판단하여 이전 blur 가져옴
        if np.max(dots_5[:,0]) - np.min(dots_5[:,0]) > 40:
            # 타원 모자이크 vectorize Complete -> 이전 타원 blur 정보 가져와서 blur처리
            for j in range(real_img.shape[0]):
                # 다 깜장색임 (타원의 일부가 없음) -> 그냥 real_img 그대로 감
                if not ellip_img[j].any():
                    continue
                else:
                    real_img[j] = [real_img[j][k] if not ellip_img[j][k] else gauss_img[j][k] for k in range(real_img.shape[1])]

            # 타원이 어긋남을 대비한 눈,코 개별 모자이크 -- 잘못 잡은 상태이니 개별 모자이크 의미 없음

            # 모자이크 위에 눈,코,귀 찍어봄 - 이미 이전 것으로 모든 blur를 마쳤으므로 ( 모든 box에 대해) continue가 아닌 break 해도 됨
            cv2.circle(real_img, center=tuple(keypoints[0][:2].astype(int)), radius=1, color=(255, 150, 205),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[1][:2].astype(int)), radius=1, color=(205, 100, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[2][:2].astype(int)), radius=1, color=(0, 190, 225),
                       thickness=-1)
            # 왼쪽 귀 - 보라? 핑크?
            cv2.circle(real_img, center=tuple(keypoints[3][:2].astype(int)), radius=1, color=(255, 0, 255),
                       thickness=-1)
            # 오른쪽 귀 - 빨강
            cv2.circle(real_img, center=tuple(keypoints[4][:2].astype(int)), radius=1, color=(0, 0, 255), thickness=-1)

            cv2.circle(real_img, center=tuple(keypoints[5][:2].astype(int)), radius=1, color=(255, 0, 0), thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[11][:2].astype(int)), radius=1, color=(100, 255, 100),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[13][:2].astype(int)), radius=1, color=(255, 255, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[7][:2].astype(int)), radius=1, color=(255, 0, 0),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[8][:2].astype(int)), radius=1, color=(0, 255, 0),
                       thickness=-1)
            break

########################################   정상적으로 lmd, rmd 찾아서 타원 근사하기 !  #########################################

        # lmd로서 "오른쪽 귀"를 잡음
        lmd_x, lmd_y = dots_5[4]
        # lmd 점과 가장 거리가 먼 점의 index를 구한다################################################


        for kk in range(5):
            # 해당 점이 lmd보다 더 오른쪽에 있으면 rmd 후보에 넣음
            if dots_5[kk][0] > lmd_x:
                dots_5_dist = np.append(dots_5_dist, np.append(dots_5[kk],kk))

        #오른쪽 귀보다 오른쪽에 아무 점도 없는 경우 --> 그냥 blur안함
        if dots_5_dist.shape[0] == 0:
            cv2.circle(real_img, center=tuple(keypoints[0][:2].astype(int)), radius=1, color=(255, 150, 205),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[1][:2].astype(int)), radius=1, color=(205, 100, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[2][:2].astype(int)), radius=1, color=(0, 190, 225),
                       thickness=-1)
            # 왼쪽 귀 - 보라? 핑크?
            cv2.circle(real_img, center=tuple(keypoints[3][:2].astype(int)), radius=1, color=(255, 0, 255),
                       thickness=-1)
            # 오른쪽 귀 - 빨강
            cv2.circle(real_img, center=tuple(keypoints[4][:2].astype(int)), radius=1, color=(0, 0, 255), thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[5][:2].astype(int)), radius=1, color=(255, 0, 0),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[11][:2].astype(int)), radius=1, color=(100, 255, 100),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[13][:2].astype(int)), radius=1, color=(255, 255, 255),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[7][:2].astype(int)), radius=1, color=(255, 0, 0),
                       thickness=-1)
            cv2.circle(real_img, center=tuple(keypoints[8][:2].astype(int)), radius=1, color=(0, 255, 0),
                       thickness=-1)
            continue
        # 오른쪽 귀보다 오른쪽에 뭐라도 있으면, 그 중 최대 거리인 점을 선택하여 rmd로 설정
        else:
            dots_5_dist = dots_5_dist.reshape((dots_5_dist.shape[0] // 3, 3))
            # lmd와 각 점까지의 거리가 float형태로 담겨있다.
            dots_5_dist = (dots_5_dist - np.array([lmd_x, lmd_y,0])) ** 2
            dots_5_dist = np.hstack([ (dots_5_dist[:,0] + dots_5_dist[:,1]).reshape(dots_5_dist.shape[0],1), dots_5_dist[:,2].reshape(dots_5_dist.shape[0],1)])
            rmd_idx = int((dots_5_dist[np.argmax(dots_5_dist[:,0])][1]) **0.5)
            rmd_x, rmd_y = dots_5[rmd_idx]
            danchuk_r = (np.max(dots_5_dist[:,0]) **0.5 )* 85 / 200
            # 위의 두 점을 이용해 얼굴이 기울어진 각도를 구함 (arctan 이용 )
            angle = int(np.arctan((rmd_y - lmd_y) / (rmd_x - lmd_x)) * 180 // np.pi)
            ############### ellip_img_former 사용하는 대신 여기서 ( 이전 정보 필요없어졌을 때 ) ellip_img를 초기화해준다. ##########
            ellip_img = np.zeros((real_img.shape[0], real_img.shape[1]))
            cv2.ellipse(ellip_img, (int((lmd_x + rmd_x) / 4 + nose_x / 2), int((lmd_y + rmd_y) / 4 + nose_y / 2)),
                        (int(danchuk_r), int(danchuk_r * 1.5)), angle, 0, 360, 255, -1)
            ####ellip_img_former = copy.deepcopy(ellip_img)


        # 그린 타원에 따라 모자이크
        for j in range(real_img.shape[0]):
            # 다 깜장색임 (타원의 일부가 없음) -> 그냥 real_img 그대로 감
            if not ellip_img[j].any():
                continue
            else:
                real_img[j] = [real_img[j][k] if not ellip_img[j][k] else gauss_img[j][k] for k in range(real_img.shape[1])]

        # 타원이 어긋남을 대비한 눈,코 개별 모자이크 -- 지금은 초록색으로 눈 찍고 모자이크한 상태 - 없애고싶으면 cv2.circle만 지우면됨
        # 모서리쪽에서 다른 얼굴이 잡히는 경우가 있으니, image width,height고려해서 처리해줘야함
        for jj in range(3):
            now_x, now_y = keypoints[jj][:2].astype(int)
            lll, rrr, uuu, ddd = now_x - 3, now_x + 4, now_y - 3, now_y + 4
            lll = now_x - 3 if now_x > 3 else 0
            rrr = now_x + 4 if now_x + 4 <= real_img_width else real_img_width
            uuu = now_y - 3 if now_y > 3 else 0
            ddd = now_y if now_y + 4 <= real_img_height else real_img_height
            real_img[uuu: ddd, lll: rrr] = cv2.GaussianBlur(real_img[uuu: ddd, lll: rrr], (9, 9), 300,300)
            # 다음 frame에서 사용해야할 때,  개별 모자이크 정보도 포함이 되어야하므로 그려줌
            ellip_img = cv2.rectangle(ellip_img, (lll, uuu), (rrr,ddd ), 255, -1)

        # 모자이크 위에 점 찍어봄
        cv2.circle(real_img, center=tuple(keypoints[0][:2].astype(int)), radius=1, color=(255, 150, 205),
                   thickness=-1)
        cv2.circle(real_img, center=tuple(keypoints[1][:2].astype(int)), radius=1, color=(205, 100, 255),
                   thickness=-1)
        cv2.circle(real_img, center=tuple(keypoints[2][:2].astype(int)), radius=1, color=(0, 190, 225),
                   thickness=-1)
        # 왼쪽 귀 - 보라? 핑크?
        cv2.circle(real_img, center=tuple(keypoints[3][:2].astype(int)), radius=1, color=(255, 0, 255),
                   thickness=-1)
        # 오른쪽 귀 - 빨강
        cv2.circle(real_img, center=tuple(keypoints[4][:2].astype(int)), radius=1, color=(0, 0, 255), thickness=-1)
        cv2.circle(real_img, center=tuple(keypoints[5][:2].astype(int)), radius=1, color=(255, 0, 0), thickness=-1)
        cv2.circle(real_img, center=tuple(keypoints[11][:2].astype(int)), radius=1, color=(100, 255, 100), thickness=-1)
        cv2.circle(real_img, center=tuple(keypoints[13][:2].astype(int)), radius=1, color=(255, 255, 255), thickness=-1)
        cv2.circle(real_img, center=tuple(keypoints[7][:2].astype(int)), radius=1, color=(255, 0, 0),
                   thickness=-1)
        cv2.circle(real_img, center=tuple(keypoints[8][:2].astype(int)), radius=1, color=(0, 255, 0),
                   thickness=-1)

    real_img = cv2.cvtColor(real_img, cv2.COLOR_RGB2BGR)

    cv2.imwrite('why/' + image_list[i], real_img)

# 생성한 이미지를 하나씩 보기 위한 프로그램

# m누르면 다음 이미지
# n 누르면 이전 이미지
# q 누르면 종료

import cv2
import numpy as npq
import os
import matplotlib.pyplot as plt

## 보고싶은 이미지들이 들어있는 경로 입력 --   !!이미지 경로는 여기만 수정하면 됨!!
file_loc = ''




image_list = os.listdir(file_loc)
# 이미지를 이름 순으로 봐야하기때문에 정렬
image_list.sort()

i = 1
# gdr_2 에서는 맥에서의 .DS_STORE를 제거하기 위해 1부터
maxi_len = len( image_list)
while True:

    img_name = file_loc + '/' + image_list[i]
    img = cv2.imread(file_loc + '/' + image_list[i])
    cv2.namedWindow("frame")  # create a named window
    cv2.moveWindow("frame", 30, 30)
    cv2.imshow("frame", cv2.resize(img, ( img.shape[1] * 2,img.shape[0]*2)))

    now_key = cv2.waitKey() & 0xFF

    if now_key == ord('n') or now_key == ord('ㅜ') or now_key == ord('N'):  # n을 누르면 뒤로 돌리기 --> 한글로 카보드 입력이 설정되어있으면, 정방향 재생 됨. 한/영 키 누르면 된다
        i = i -1 if i>0 else 0
        cv2.destroyAllWindows()
        continue

    # 다음 이미지로 넘기기
    elif now_key == ord('m') or now_key == ord('ㅡ') or now_key == ord('M'):
        i = i + 1 if i < maxi_len - 1 else i
        cv2.destroyAllWindows()
        continue
    elif now_key == ord('s'):
        cv2.imwrite(file_loc+'_error/' + image_list[i], img)
    # for analysis
    elif now_key == ord('a'):
        cv2.imwrite(file_loc+'_analysis/' + image_list[i], img)
    elif now_key == ord('q'):
        cv2.destroyAllWindows()
        break
    # 나머지는 모두 현 이미지 유지
    else:
        pass


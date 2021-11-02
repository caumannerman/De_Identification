# De_Identification

해당 프로그램은 "원본 Image Input -- rcnn-resnet50 모델로 얼굴,관절 Keypoint 추출 -- 얼굴 비식별화"의 과정을 거쳐 나온 Output 으로서, 코드는 Python, Pytorch, opencv, Pillow 라이브러리를 사용했습니다.
옆모습, 뒷모습 모두 하나의 알고리즘으로 비식별화를 가능하게 설계하였고, 
뒷모습인 경우에는 불필요한 비식별화를 하지 않도록 알고리즘을 설계하였습니다.

Pytorch rcnn-resnet50 관련 정보: https://learnopencv.com/human-pose-estimation-using-keypoint-rcnn-in-pytorch/

사용한 resnet50의 keypoint추출 17개

<img width="744" alt="스크린샷 2021-11-03 오전 12 19 58" src="https://user-images.githubusercontent.com/75043852/139877013-7d94820a-5420-4f01-b2f6-74a57c47f57e.png">


비식별화 결과 


<img width="949" alt="스크린샷 2021-10-31 오후 2 39 22" src="https://user-images.githubusercontent.com/75043852/139574252-52f9e647-8b5f-435e-99a2-119c16c67aca.png">
<img width="955" alt="연구대회메인" src="https://user-images.githubusercontent.com/75043852/139574284-6cc9c986-bde9-4d66-8c16-b0e639aa7cd0.png">
<img width="949" alt="스크린샷 2021-10-31 오후 2 39 22" src="https://user-images.githubusercontent.com/75043852/139574304-ed03c472-5451-43c1-ae73-21866891f06b.png">
<img width="955" alt="캡스톤경진대회메인" src="https://user-images.githubusercontent.com/75043852/139574309-90b21fca-7340-489e-b9d7-41b476f73c47.png">

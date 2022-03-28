# 킥라니 멈춰!

## 데이터 청년 캠퍼스 고려대과정 8조 

<img width="283" alt="킥라니멈춰(오리지널)" src="https://user-images.githubusercontent.com/60602927/131216960-52f75fd4-68cb-427e-af94-bfb9e89f1125.png">
         
## Introduction
전동킥보드는 편리하게 사용할 수 있다는 장점을 가졌지만 점차 사용자가 많아지면서, 전동킥보드 교통사고 건수도 매해  2배씩 증가하고 있습니다. 이에 따라 도로교통법이 강화되었으나, 여전히 공유킥보드에 대해 안전 수칙을 지키는 사용자들은 미미한 상황입니다. 따라서 저희는 이러한 공유킥보드의 **안전수칙 위반 문제를 해결**하고, 보다 **안전한 사용 문화를 확립**하기 위한 서비스를 구현해보았습니다.  
 
## Project Explanation
저희는 이러한 공유킥보드의 안전수칙 위반 문제를 해결하기 위한 방안으로 다음과 같은 핵심 기능을 구현하였습니다.
- **사용자 인식**: 카메라에 사용자가 있는지 확인. 사용자가 인식되지 않는 경우, 얼굴을 보여달라는 문구 표시.
- **헬멧 탐지** : 헬멧 착용 여부 탐지 후, 미착용 상태가 지속되면 경고 문구 표시 및 음성 알림.    
다양한 형태의 헬멧 탐지 가능 / 헬멧과 모자 구분 가능.
- **2인이상 동반 탑승 탐지**: 탑승 인원 탐지 후 화면에 표시. 탑승 인원 수칙 위반이 일정시간 이상 지속되면 경고 문구 표시.
- **위험 자세 인식**: 한손 운전, 양손을 놓고 타는지 여부, 휴대폰을 보며 고개를 숙이는 위험 자세 탐지.    
위험 자세가 포착되면 경고 문구 알림.

![image](https://user-images.githubusercontent.com/60602927/131278708-a0d4142c-e85d-4a32-9dad-69b8bdba9ac7.png)


## Project UI
### Title
KickRani_Stop

### Version
0.1.1

### Date/Publication
2021-08-30 15:52:26 (GMT+9)

### Service Implement
직접 제작한 UI를 통해 실시간 탐지를 해본 내용입니다.  

![image](https://user-images.githubusercontent.com/60602927/131379225-e4f086a6-3c22-47c9-8032-1803f2afba5f.png)
![image](https://user-images.githubusercontent.com/60602927/131278883-9104d4ec-a8e3-4a81-b35d-4464783e5a1d.png)
![image](https://user-images.githubusercontent.com/60602927/131381120-83004562-e2d4-4936-8a31-b0476ae87392.png)



## 서비스 활용방안 & 기대효과 

![image](https://user-images.githubusercontent.com/60602927/131279084-cd78a36d-e0b4-4445-9d29-cf6abcaf228f.png)
![image](https://user-images.githubusercontent.com/60602927/131430920-fd61f20f-f9a3-4388-8260-686107045ee7.png)
![image](https://user-images.githubusercontent.com/60602927/131430721-47d78fe4-3d6a-4ff1-b4e7-a06ce8d0c243.png)
![image](https://user-images.githubusercontent.com/60602927/131431099-89f32bf2-8dc0-4d48-bc55-4f47a84bb563.png)

## Team 킥라니 멈춰 ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<table>
  <tr>
      <td align="center"><a href="https://github.com/dirn5454"><img src= "https://user-images.githubusercontent.com/60602927/131469094-46aaf5f6-261c-4a05-a189-2744a4c657e7.jpg" width="120px;" alt=""/><br /><sub><b>Jongwoong Lee</b></sub></a><br /><a href="#design-tbenning" title="Design">🎨</a> <a href="#code" title="Code">💻</a> <a href="#tool-jfmengels" title="Tools">🔧</a> <a href="#content" title="Content">🖋</a></td>
      <td align="center"><a href="https://github.com/HaeSung-Oh"><img src= "https://user-images.githubusercontent.com/60602927/131469843-ee38133b-3a35-403c-b072-3e5bd708e796.png" width="120px;" alt=""/><br /><sub><b>Haesung Oh</b></sub></a><br /><a href="#eventorganizing" title="Event Organizing">💡</a> <a href="#code" title="Code">💻</a> <a href="#tool-jfmengels" title="Tools">🔧</a> </a> <a href="#content" title="Content">🖋</a></td>
    <td align="center"><a href="https://github.com/vndrudkim21"><img src= "https://user-images.githubusercontent.com/60602927/131469559-84949cd7-7df4-46cf-8b91-65570dcbbb6c.png" width="120px;" alt=""/><br /><sub><b>Seongju Kim</b></sub></a><br /><a href="#talk-kentcdodds" title="Talks">📢</a> <a href="#code" title="Code">💻</a> <a href="#tool-jfmengels" title="Tools">🔧</a> <a href="https://github.com/all-contributors/all-contributors/commits?author=kentcdodds" title="Documentation">📖</a> </td>
    <td align="center"><a href="https://github.com/youngsin1108"><img src="https://user-images.githubusercontent.com/60602927/131469625-ecc30b49-2517-4fbb-a401-cc5d9550f2f5.png" width="120px;" alt=""/><br /><sub><b>Youngsin Lee</b></sub></a><br /> <a href="#question-kentcdodds" title="Answering Questions">💬</a> <a href="#code" title="Code">💻</a> <a href="#eventorganizing" title="Event Organizing">💡 </a> <a href="#tool-jfmengels" title="Tools">🔧</a> </td>
    <td align="center"><a href="https://github.com/minchan-byun"><img src="https://user-images.githubusercontent.com/60602927/131470046-30939439-2c06-4d88-a679-fa89c791f725.png" width="120px;" alt=""/><br /><sub><b>Minchan Byun</b></sub></a><br /><a href="#question-kentcdodds" title="Answering Questions">💬</a> <a href="#projectmanagment" title="Project Managemnt">📆 <a href="#code" title="Code">💻 <a href="#tool-jfmengels" title="Tools">🔧</a> </td>
  </tr>
</table>

import cv2 as cv2

def save(save_name):

  # 選擇第二隻攝影機
  cap = cv2.VideoCapture(0)
  # 設定擷取影像的尺寸大小
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

  # 使用 XVID 編碼
  fourcc = cv2.VideoWriter_fourcc(*'mp4v')

  # 建立 VideoWriter 物件，輸出影片至 output.avi
  # FPS 值為 20.0，解析度為 640x360
  out = cv2.VideoWriter(save_name, fourcc, 20.0, (640, 480))

  while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
      # 寫入影格
      out.write(frame)
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    else:
      break


  while(True):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()

    # 顯示圖片
    cv2.imshow('frame', frame)

    # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # 釋放攝影機
  cap.release()

  # 關閉所有 OpenCV 視窗
  cv2.destroyAllWindows()



def camara(bx,by,bw,bh):
  cap = cv2.VideoCapture(0)

  while(True):
    # 從攝影機擷取一張影像
    ret, frame = cap.read()
    cv2.rectangle(frame, (int(bx+bw/2), int(by+bh/2)), (int(bx-bw/2), int(by-bh/2)), (0, 255, 0), 2)
    # 顯示圖片
    cv2.imshow('frame', frame)
    # 若按下 q 鍵則離開迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  # 釋放攝影機
  cap.release()

  # 關閉所有 OpenCV 視窗
  cv2.destroyAllWindows()



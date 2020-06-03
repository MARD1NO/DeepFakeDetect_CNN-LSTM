import os
import shutil

if __name__ == "__main__":
    """
    删除少于20张图片的文件夹
    """
    frameImageDir = '/home/aistudio/work/validate_face_image'
    print(len(os.listdir(frameImageDir)))
    frameImageDirLists = os.listdir(frameImageDir)
    cnt = 0
    for frameImageDirList in frameImageDirLists:
        frameFullDir = os.path.join(frameImageDir, frameImageDirList)
        # print(frameFullDir)
        if len(os.listdir(frameFullDir)) < 10:
            # print("???")
            print(frameFullDir)

            cnt += 1
            shutil.rmtree(frameFullDir)
            
    print(cnt)

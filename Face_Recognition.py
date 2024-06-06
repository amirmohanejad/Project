#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2


# In[2]:


#بارگذاری فایل xml
xml = r".\xml_file\haarcascade_frontalface_default.xml"
#xml = r"C:\Users\arya2\OneDrive\Desktop\AmirMahdiMohamadiNejad\xml_file\haarcascade_frontalface_default.xml"

#کلاس لازم برای تشخیص چهره از کتابخانه cv2
xmlf = cv2.CascadeClassifier(xml)
#رزرو کردن وبکم اصلی لپتاپ من که آیدیش 0 هست
cam = cv2.VideoCapture(0)
#شمارنده برای سیو عکس
i = 0
#اگه وبکم بدون مشکل باز شد
if cam.isOpened():
    #حلقه بینهایت اینجا یعنی با سرعت عکس بگیره از وبکم که مجموعه این عکسا فریم های فیلم نمایشی از دوربین رو تشکیل میدن
    while True:
        #تصویر از وبکم گرفته بشه و بصورت یک ماتریس سه بعدی ذخیره بشه و اگه نتیجه عملیات تو متغیر رت ذخیره بشه 
        ret, img = cam.read()
        #چون کتابخانه اپن-سیوی فقط قادر به تشخیص اشیا در تصاویر سیاه و سفید هست تصویر رو برای درک اون به سیاه سفید تبدیل میکنیم
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = xmlf.detectMultiScale(
            #تصویر سیاه سفید
            gray,
            #معیار تعادل بزرگی کوچکی تشخیص چهره
            scaleFactor = 1.15,
            #مشخص میکنه به چند پنجره نزدیک پنجره اصلی برای تشخیص چهره توجه بشه
            minNeighbors =5,
            #اندازه پنجره ها
            minSize = (30, 30)
        )
        #اگه چهره ای تشخیص داد تعداد تشخیص داده شده رو چاپ کنه و در غیر اینصورت پیغام چیزی پیدا نشد چاپ کنه
        found = int(format(len(faces)))
        if found > 0:
            print("Found {0} faces".format(len(faces)))
        else:
            print("No face found!")
        #عملیات رسم مستطیل دور چهره روی تصویر اصلی
        #نقطه عرضی a
        #نقطه طولی b
        #اندازه عرض c
        #اندازه طول d
        for (a, b, c, d) in faces:
            cv2.rectangle(img, (a, b), (a+c, b+d), (0, 255, 0), 2)
        cv2.imshow("تشخیص چهره", img)
        #اینجا 1 زمان گرفتن هر عکس فریم است
        x = cv2.waitKey(1)
        #دکمه خروج
        if x == ord('q'):
            break
        #دکمه سیو آخرین فریم گرفته شده
        elif x == ord('s'):
            cv2.imwrite(f"pic_{i}.png", img)
            i+=1
#اگه باز کردن وبکم با مشکل (مثل دادن آیدی اشتباه) همراه بود پیغام مناسب نمایش داده شود
else:
    print("Camera is not working properly!")
#بستن درخواست دوربین تا اگه نرم افزار دیگه ای نیازش داشت به اون داده بشه
cam.release()
#بستن تمام پنجره ها
cv2.destroyAllWindows()

#پروژه امیر مهدی محمدی نژاد


หาขอบเขตของหมู่บ้านจาก latitude และ longitude

* model หลักที่ใช้คือ zero shot learning โดยต้องใช้ pytorch
* clustering
  * DBSCAN
  * Hierarchy clustering

Directory

* Data 
  * เก็บข้อมูลจากกรมที่ดิน และ ข้อมูลที่เกี่ยวข้อง
* notebook
  * สำหรับอธิบายแบ่งเป็น 3 part คือ 1 -2 -3
  * สำหรับใช้รันและทดลองชื่อ experiment โดยจะเป็น notebook หลัก สามารถใช้ notebook นี้รันได้
* script_file
  * py file สำหรับใช้จัดการข้อมูลโดยมี function อยู่ภายใน
  
  
การพัฒนาต่อยอด

1. train หรือ หา custom distance ที่เหมาะสมใหม่
2. model เพื่อใช้ในการ embed image และ filter
3. input สำหรับ cluster สามารถเพิ่ม feature อื่นได้
4. วิธีในการตีกรอบ polygon นอกจาก convexhull ที่ใช้ในการทดลองนี้
5. polygon ที่เป็น output สามารถนำมาต่อยอดได้ เช่น union เป็นต้น
6. config ค่าที่เหมาะสมกับงาน

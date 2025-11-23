import paramiko
import os
import stat  # Dosya tipi (dizin mi?) kontrolü için

"""
    Bu betik HPC makinasında kaydedilen modellerin indirilmesini sağlar
"""

# --- HPC Bağlantı Bilgileriniz ---
HPC_HOST = "172.16.6.11"
HPC_PORT = 22
HPC_KULLANICI = "XXXX"
HPC_SIFRE = "YYYY"

# --- Dosya Yolları ---
UZAK_DIZIN = f"/arf/scratch/{HPC_KULLANICI}/CitationIntentClassifier/checkpoints_v1"
YEREL_DIZIN = "./checkpoints_v1_indirilen"


# ----------------------------------

def sftp_indir_recursive(sftp, uzak_yol, yerel_yol):
    """
    Paramiko SFTP kullanarak uzak sunucudaki bir dizini özyinelemeli olarak indirir.
    """
    print(f"Yerel dizin oluşturuluyor: {yerel_yol}")
    os.makedirs(yerel_yol, exist_ok=True)

    print(f"Uzak dizin listeleniyor: {uzak_yol}")
    for item in sftp.listdir_attr(uzak_yol):
        # Dosya/dizin adlarını birleştir
        uzak_item_yolu = os.path.join(uzak_yol, item.filename)
        yerel_item_yolu = os.path.join(yerel_yol, item.filename)

        # Öğenin bir dizin mi yoksa dosya mı olduğunu kontrol et
        if stat.S_ISDIR(item.st_mode):
            # Bu bir DİZİN: Fonksiyonu tekrar çağır
            print(f"  Alt dizine giriliyor: {uzak_item_yolu}")
            sftp_indir_recursive(sftp, uzak_item_yolu, yerel_item_yolu)
        else:
            # Bu bir DOSYA: İndir
            if os.path.exists(yerel_item_yolu):
                print(f"  Dosya zaten var, atlanıyor: {yerel_item_yolu}")
            else:
                # Dosya yerelde yoksa indir
                print(f"  Dosya indiriliyor: {uzak_item_yolu} -> {yerel_item_yolu}")
                sftp.get(uzak_item_yolu, yerel_item_yolu)


# --- Ana Bağlantı Kodu ---
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    print(f"{HPC_HOST} adresine bağlanılıyor...")
    # Şifre ile bağlanma
    ssh.connect(HPC_HOST, port=HPC_PORT, username=HPC_KULLANICI, password=HPC_SIFRE)

    # (SSH Anahtarı ile bağlanmak isterseniz üst satırı yorumlayıp alt satırı kullanın)
    # anahtar_yolu = "/Users/kullaniciniz/.ssh/id_rsa"
    # ssh.connect(HPC_HOST, port=HPC_PORT, username=HPC_KULLANICI, key_filename=anahtar_yolu)

    print("Bağlantı başarılı. SFTP oturumu açılıyor...")
    sftp = ssh.open_sftp()

    # Özyinelemeli indirme fonksiyonunu başlat
    sftp_indir_recursive(sftp, UZAK_DIZIN, YEREL_DIZIN)

    print("\nİşlem tamamlandı!")
    print(f"Uzak dizin: {UZAK_DIZIN}")
    print(f"Yerel hedef: {YEREL_DIZIN}")

except paramiko.AuthenticationException:
    print("Hata: Kimlik doğrulama başarısız! Kullanıcı adı veya şifrenizi kontrol edin.")
except Exception as e:
    print(f"Beklenmedik bir hata oluştu: {e}")

finally:
    if 'sftp' in locals():
        sftp.close()
    if 'ssh' in locals():
        ssh.close()
    print("Bağlantılar kapatıldı.")
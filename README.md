# ResNet GAN untuk CelebA

Repositori ini melatih GAN tanpa label untuk menghasilkan wajah berukuran 128×128 piksel dari dataset CelebA. Generator dan critic dibangun dari residual block; generator melakukan upsampling dari vektor laten 128 dimensi, sedangkan critic melakukan downsampling dengan spectral normalization. Training memakai hinge adversarial loss dan memperbarui generator setiap dua langkah critic.

## Struktur singkat

| Berkas | Fungsi |
| --- | --- |
| `main.py` | Entry point training, pengaturan seed, device, direktori output, dan model. |
| `train.py` | Loop optimisasi critic dan generator serta penyimpanan sampel/checkpoint. |
| `models.py` | Definisi generator, critic, dan residual block. |
| `data.py` | Mengunduh CelebA dan membuat `DataLoader`. |
| `infer.py` | Memuat checkpoint generator dan membuat grid 64 gambar. |
| `constant.py` | Hyperparameter dan lokasi data/output. |
| `utils.py` | Penyimpanan sampel dan checkpoint. |

## Persyaratan

- Python 3.13 atau lebih baru.
- [`uv`](https://docs.astral.sh/uv/) untuk memasang dependensi dari `pyproject.toml` dan `uv.lock`.
- GPU CUDA bersifat opsional. Urutan pemilihan device adalah CUDA, MPS, lalu CPU.
- Ruang penyimpanan dan koneksi internet untuk mengunduh CelebA pada eksekusi pertama. CelebA ditujukan untuk penggunaan riset nonkomersial; baca [ketentuan resminya](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) sebelum mengunduh atau mendistribusikan data.

Pasang dependensi dari direktori repositori:

```bash
uv sync
```

## Training

1. Sesuaikan konfigurasi di `constant.py` bila perlu.
2. Jalankan training:

   ```bash
   uv run python main.py
   ```

Dataset akan diunduh ke `data/`. Sampel disimpan di `outputs/`, sedangkan checkpoint disimpan setiap lima epoch di `checkpoints/`.

## Inferensi

Jalankan inferensi dengan checkpoint hasil training:

```bash
uv run python infer.py \
  --checkpoint-path checkpoints/wgan_gp_0050.pt \
  --output-path outputs/inference.png \
  --seed 42
```

Argumen pendek:

```bash
uv run python infer.py -cp checkpoints/wgan_gp_0050.pt -o outputs/inference.png
```

Jika `--output-path` tidak diberikan, hasil disimpan sebagai `outputs/generated_seed_<seed>.png`.

## Referensi

### Model dan komponen

1. I. Goodfellow et al., “Generative Adversarial Nets,” *NeurIPS*, 2014. [arXiv:1406.2661](https://arxiv.org/abs/1406.2661).
2. A. Radford, L. Metz, dan S. Chintala, “Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks,” *ICLR*, 2016. [arXiv:1511.06434](https://arxiv.org/abs/1511.06434).
3. K. He, X. Zhang, S. Ren, dan J. Sun, “Deep Residual Learning for Image Recognition,” *CVPR*, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
4. T. Miyato, T. Kataoka, M. Koyama, dan Y. Yoshida, “Spectral Normalization for Generative Adversarial Networks,” *ICLR*, 2018. [arXiv:1802.05957](https://arxiv.org/abs/1802.05957).
5. S. Ioffe dan C. Szegedy, “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift,” *ICML*, 2015. [arXiv:1502.03167](https://arxiv.org/abs/1502.03167).
6. S. Elfwing, E. Uchibe, dan K. Doya, “Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning,” *Neural Networks*, 2018. [arXiv:1702.03118](https://arxiv.org/abs/1702.03118).
7. D. P. Kingma dan J. Ba, “Adam: A Method for Stochastic Optimization,” *ICLR*, 2015. [arXiv:1412.6980](https://arxiv.org/abs/1412.6980).

### Dataset

8. Z. Liu, P. Luo, X. Wang, dan X. Tang, “Deep Learning Face Attributes in the Wild,” *ICCV*, 2015. [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

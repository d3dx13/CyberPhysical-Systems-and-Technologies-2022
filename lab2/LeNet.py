import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

IMAGE_SIZE = (28, 28)
OUTPUT_MIN = np.array([-12.405435, -10.730108, -12.911035, -12.028286, -11.388719,
                       -11.590592, -18.247143, -13.193559, -7.603819, -10.83747, ])
OUTPUT_MAX = np.array([12.5716915, 16.748579, 19.614042, 19.389887, 18.39213,
                       18.968506, 17.291187, 17.354427, 18.150793, 17.49377, ])
CONFIDENCE_THRESHOLD = 0.7643010914325714


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # convolutional layers
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=5 // 2).cuda(),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0).cuda(),
            nn.Sigmoid(),
            nn.AvgPool2d(kernel_size=(2, 2), stride=2),
        )

        self.linear_layers = 16 * 5 * 5

        # linear layers
        self.fc = nn.Sequential(
            nn.Linear(self.linear_layers, 120).cuda(),
            nn.Sigmoid(),
            nn.Linear(120, 84).cuda(),
            nn.Sigmoid(),
            nn.Linear(84, 10).cuda(),
        )

        self.printed_size = False

    def forward(self, x):
        # convolutional layers with ReLU and pooling
        x = self.conv(x)

        if not self.printed_size:
            print(x.size())
            self.printed_size = True
        x = x.flatten(start_dim=1)

        # linear layers
        x = self.fc(x)
        return x


if __name__ == '__main__':
    from importlib import reload
    import cv2
    import torchvision.transforms as transforms

    normalize_transforms = transforms.Compose([
        transforms.Resize(size=IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.Normalize((0.5,), (0.5,)),
    ])


    def _remove_attached_to_image_border(image):
        pad = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=255)
        h, w = pad.shape

        # create zeros mask 2 pixels larger in each dimension
        mask = np.zeros([h + 2, w + 2], np.uint8)

        # floodfill outer white border with black
        img_floodfill = cv2.floodFill(pad, mask, (0, 0), 0, (5), (0), flags=8)[1]

        # remove border
        img_floodfill = img_floodfill[1:h - 1, 1:w - 1]

        return img_floodfill


    model_new = LeNet()
    model_new.cuda()
    model_new.load_state_dict(torch.load(f'LeNet_archive.pt'))

    camera = cv2.VideoCapture(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    while (True):

        ret, image = camera.read()

        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame = frame[:, (frame.shape[1] - frame.shape[0]) // 2:]
        frame = frame[:, :frame.shape[0]]

        frame = cv2.medianBlur(frame, 11)

        ret, thresh = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        thresh = cv2.dilate(thresh, kernel, iterations=1)
        thresh = cv2.erode(thresh, kernel, iterations=1)

        thresh = _remove_attached_to_image_border(thresh)
        components = cv2.connectedComponentsWithStats(thresh, 8, cv2.CV_32S, )
        components_stats = components[2].tolist()
        for i in range(len(components_stats)):
            components_stats[i].append(i)
        components_stats = sorted(components_stats, key=lambda x: x[4], reverse=True)
        thresh = np.zeros_like(thresh)
        thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        if len(components_stats) > 1 and components_stats[1][-2] > thresh.shape[0] * thresh.shape[1] * 0.01:
            thresh[components[1] == components_stats[1][-1]] = 255
            thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

            thresh = np.reshape(thresh, (1, 1, thresh.shape[0], thresh.shape[1]))

            data = normalize_transforms(torch.tensor(np.float32(thresh) / 255.0).cuda())
            data_cpu = data.detach().cpu().numpy()

            output = np.divide(model_new(data).detach().cpu().numpy() - OUTPUT_MIN, OUTPUT_MAX - OUTPUT_MIN)
            output_argmax = np.argmax(output)
            output_max = np.max(output)

            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (0, thresh_bgr.shape[0] // 2 + 50)
            fontScale = 5
            thickness = 10
            if output_max > CONFIDENCE_THRESHOLD:
                color = (0, 255, 0)
            else:
                color = (0, 0, 127)
            thresh_bgr = cv2.putText(thresh_bgr, str(output_argmax), org, font,
                                fontScale, color, thickness, cv2.LINE_AA)
            image = cv2.putText(image, str(output_argmax), org, font,
                                fontScale, color, thickness, cv2.LINE_AA)

        cv2.imshow('thresh', thresh_bgr)
        cv2.imshow('image', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

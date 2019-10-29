from config.config_Faster_RCNN import cfg
from models.Faster_RCNN import Faster_RCNN
import time

if __name__ == '__main__':
    net = Faster_RCNN('Faster_RCNN', 'resnet101', True)

    for epoch in range(cfg.epochs):
        loss = 0
        acc = 0
        for step in range(int(cfg.train_num / cfg.batch_size)):
            start_time = time.time()
            if step % 5000 == 0 and step != 0:
                # global_step, train_loss, train_acc, lr = net.train(True)
                global_step, rpn_box_loss, rpn_cls_loss, lr = net.train_rpn(True)
            else:
                # net.train(False)
                # global_step, train_loss, train_acc, lr = net.train(False)
                global_step, rpn_box_loss, rpn_cls_loss, lr = net.train_rpn(False)
            end_time = time.time()
            print('Epoch {} step {}, rpn_box_loss = {}, rpn_cls_loss = {} , processing time = {} lr = {}'.format(epoch, global_step,
                                                                                                rpn_box_loss, rpn_cls_loss,
                                                                                                end_time - start_time,
                                                                                                lr))
            # loss += train_loss
            # acc += train_acc

        # print('Epoch {} step {}, loss = {}, acc = {}'.format(epoch, global_step, loss / step, acc / step))


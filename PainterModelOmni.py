import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
from data_utils import *
from pprint import pprint
from const import COLORS, SHAPES


class InstEncoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.rnn_size = 64
        self.input_size = 64
        # https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/models/AttModel.py#L43
        self.embed = nn.Embedding(vocab_size + 1, self.input_size)
        # https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/models/AttModel.py#L372
        self.lstm_cell = nn.LSTMCell(self.input_size, self.rnn_size)

    def forward(self, inst):
        # TODO early stop when column contains all zero
        # http://pytorch.org/docs/master/nn.html#torch.nn.LSTMCell
        batch_size = inst.size(0)
        h = Variable(torch.zeros(batch_size, self.rnn_size).cuda())
        c = Variable(torch.zeros(batch_size, self.rnn_size).cuda())
        hiddens = []
        for i in range(inst.size(1)):
            if i > 1 and inst[:, i].sum().data[0] == 0:
                break
            input = self.embed(inst[:, i])
            h, c = self.lstm_cell(input, (h, c))
            hiddens.append(h)
        # only want the last hidden state
        hiddens = torch.stack(hiddens, dim=1) # B x seq x rnn_size
        output = []
        for i in range(inst.size(0)):
            ix = torch.nonzero(inst[i].data).squeeze()[-1]
            output.append(hiddens[i, ix])
        output = torch.stack(output, dim=0)
        return output


class CanvasEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_size = 4
        # self.obj_embed = nn.Linear(3, self.embed_size)

    def forward(self, canvas):
        # canvas: batch_size x 25 x 3
        # obj_embedding = self.obj_embed(canvas)
        # result = canvas[:, :, 2:]
        result = canvas
        assert result.size(2) == self.embed_size
        return result
        # return obj_embedding
        # canvas_embedding = obj_embedding.sum(1)  # batch x 64
        # return canvas_embedding

def sample_probs(probs):
    dist = Categorical(probs)
    sample = dist.sample()
    log_prob = dist.log_prob(sample)
    return sample.data, log_prob

def canvas2grid(canvas, h=5, w=5):
    grid = [["[ ]" for i in range(w)] for i in range(h)]
    for row in range(5):
        for col in range(5):
            if canvas[row * 5 + col].sum() >= 0:
                grid[row][col] = COLORS[canvas[row * 5 + col][0]][0] + '|' + SHAPES[canvas[row * 5 + col][1]][0]
    return grid


def index_join(a, b, index1, index2):
    c = a.new(a.size(0) + b.size(0))
    for i in range(index1.size(0)):
        c[index1[i]] = a[i]
    for i in range(index2.size(0)):
        c[index2[i]] = b[i]
    return c

def index_join_var(a, b, index1, index2):
    c = Variable(a.data.new(a.size(0) + b.size(0)))
    for i in range(index1.size(0)):
        c[index1[i]] = a[i]
    for i in range(index2.size(0)):
        c[index2[i]] = b[i]
    return c

class Shape2DPainterNet(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # self.inst_encoder_abs = InstEncoder(vocab_size)
        # # self.inst_encoder_rel_abs = InstEncoder(vocab_size)
        # self.inst_encoder_rel = InstEncoder(vocab_size)
        self.inst_encoder = InstEncoder(vocab_size)
        # self.rel_inst_encoder = InstEncoder(vocab_size)
        self.use_mask = True
        # self.canvas_encoder = CanvasEncoder()
        self.hidden_size = 64
        rnn_size = self.inst_encoder.rnn_size
        self.pred_act = True
        if self.pred_act:
            self.fc_act = nn.Linear(rnn_size, 2)
        self.pred_ref_type = False
        if self.pred_ref_type:
            self.fc_ref_type = nn.Linear(rnn_size, 2)
        self.fc_color = nn.Linear(rnn_size, 3)
        self.fc_shape = nn.Linear(rnn_size, 3)
        self.fc_abs_loc = nn.Linear(rnn_size, 25)
        self.fc_ref_obj = nn.Sequential(nn.Linear(68, 32), nn.ReLU(), nn.Linear(32, 1))
        self.rel_loc_p = nn.Linear(rnn_size, 8)
        # self.fc_offset = nn.Linear(self.inst_encoder.rnn_size, 2)
        # self.fc_rel_loc = nn.Sequential(nn.Linear(2, 16), nn.Linear(16, 25))
        # self.fc_rel_loc = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 25))
        # self.fc_rel_loc = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 25))
        self.rewards = None
        self.saved_log_probs = None
        self.running_baseline = 0
        self.saved_actions = None
        self.color_log_prob = None
        self.shape_log_prob = None
        self.loc_log_prob = None
        self.att_log_prob = None

    def loc_relative_predict1(self, inst_embedding, canvas, ref_obj):
        # offset = F.hardtanh(self.fc_offset(inst_embedding))  # Bx2
        offset = self.fc_offset(inst_embedding)  # Bx2
        # inst2 = torch.unsqueeze(inst_embedding, 1)  # Bx1x64
        # inst2 = inst2.repeat(1, 25, 1) # Bx25x64
        # att = torch.cat([inst2, canvas], 2) # Bx25x68
        # att = self.fc_ref_obj(att).squeeze() # Bx25
        # weight = F.softmax(att, dim=1)
        # # att_sample, self.att_log_prob = sample_probs(weight)
        # # ref_obj = canvas.data.new(canvas.size(0), 2)
        # # for i in range(canvas.size(0)):
        # #     ref_obj[i] = canvas.data[i, att_sample[i], 2:]
        # # ref_obj = Variable(ref_obj)
        # ref_obj = torch.bmm(weight.unsqueeze(1), canvas).squeeze(1)
        # return self.fc_rel_loc(offset + Variable(ref_obj.float().cuda())[:, 2:])
        return self.fc_rel_loc(offset + Variable(ref_obj[:, 2:].float().cuda()))
        # return self.fc_rel_loc(torch.cat([offset, ref_obj], dim=1))
        # return self.fc_rel_loc(offset + ref_obj[:, 2:])
        # return self.fc_rel_loc(torch.cat([offset, ref_obj[:, 2:]], dim=1))
        # return ref_obj[:, 2:] + offset
        # return self.fc_rel_loc(ref_obj[:, 2:] + offset)

    def ref_obj_att(self, inst_embedding, canvas):
        inst2 = torch.unsqueeze(inst_embedding, 1)  # Bx1x64
        inst2 = inst2.repeat(1, 25, 1) # Bx25x64
        att = torch.cat([inst2, canvas], 2) # Bx25x68
        att = self.fc_ref_obj(att).squeeze(dim=2) # Bx25
        weight = F.softmax(att, dim=1)
        att_sample, att_log_prob = sample_probs(weight)
        self.att_sample = att_sample
        self.att_weight = weight
        ref_obj = torch.LongTensor(canvas.size(0), 4)
        for i in range(canvas.size(0)):
            ref_obj[i] = canvas.data[i, att_sample[i]].long().cpu()
        return ref_obj, att_log_prob

    def loc_relative_predict(self, inst_embedding, canvas, ref_obj):
        offsets = [(-1, 0), (1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        loc_probs = F.softmax(self.rel_loc_p(inst_embedding), dim=1)
        loc_sample, loc_log_prob = sample_probs(loc_probs)
        ref_obj_att, att_log_prob = self.ref_obj_att(inst_embedding, canvas)
        att_reward = None
        if ref_obj is not None:
            att_reward = (((ref_obj_att == ref_obj).sum(dim=1) == 4).float() - 0.5) * 2
        loc_predict = ref_obj_att.new(ref_obj_att.size(0))
        for i in range(ref_obj_att.size(0)):
            offsetx, offsety = offsets[loc_sample[i]]
            # loc_predict[i] = (ref_obj[i, 2] + offsetx) * 5 + (ref_obj[i, 3] + offsety)
            loc_predict[i] = (ref_obj_att[i, 2] + offsetx) * 5 + (ref_obj_att[i, 3] + offsety)
        return loc_predict, loc_log_prob, att_log_prob, att_reward

    def loc_abs_predict(self, inst_embedding):
        loc_probs = F.softmax(self.fc_abs_loc(inst_embedding), dim=1)
        loc_sample, loc_log_prob = sample_probs(loc_probs)
        return loc_sample, loc_log_prob

    def is_abs_inst(self, inst_type, dialog_ix):
        if inst_type[0] == INST_ABS:
            return True
        return False

    def forward(self, dialog, ix_to_word, eval):
        self.color_rewards = []
        self.shape_rewards = []
        self.loc_rewards = []
        self.color_log_probs = []
        self.shape_log_probs = []
        self.loc_log_probs = []
        self.att_log_probs = []
        self.att_rewards = []
        self.target_rewards = []
        self.act_log_probs = []
        self.act_rewards = []
        # self.ref_type_samples = []
        self.rel_ref_indices = []
        if self.pred_ref_type:
            self.ref_type_log_probs = []
            self.ref_type_rewards = []
        total_steps = len(dialog)
        # total_steps = 1
        final_canvas = dialog[-1][1].long()
        init_canvas = dialog[0][6].long()
        canvas_updated = init_canvas.clone()
        running_reward = torch.LongTensor(final_canvas.size(0)).fill_(1)
        for dialog_ix in range(total_steps):
            # get data
            inst, current_canvas, target, ref, _, inst_type, prev_canvas, gt_act = dialog[dialog_ix]
            # if dialog_ix > 0:
            #     prev_canvas = Variable(dialog[dialog_ix - 1][1].cuda())
            inst_str = [' '.join(map(ix_to_word.get, list(inst[ix]))) for ix in range(inst.size(0))]
            # inst_type in a batch much be the same
            # assert ((inst_type - inst_type[0]) == 0).all()
            #
            # for print_ix in range(10):
            #     print(inst_str[print_ix])
            #     pprint(canvas2grid(final_canvas[print_ix]))
            #     print(target[print_ix])
            #     print("=================================")

            inst_embedding = self.inst_encoder(Variable(inst.cuda()))  # Bx64

            # if self.is_abs_inst(inst_type, dialog_ix):
            #     inst_embedding = self.inst_encoder_abs(Variable(inst.cuda()))  # Bx64
            # else:
            #     inst_embedding = self.inst_encoder_rel(Variable(inst.cuda()))  # Bx64

            # # encode instruction
            # if ref is None:
            #     inst_embedding = self.inst_encoder(Variable(inst.cuda()))  # Bx64
            # else:
            #     inst_embedding = self.rel_inst_encoder(Variable(inst.cuda()))  # Bx64
            if self.pred_ref_type:
                ref_type_probs = F.softmax(self.fc_ref_type(inst_embedding), dim=1)
                ref_type_sample, ref_type_log_prob = sample_probs(ref_type_probs)
                self.ref_type_log_probs.append(ref_type_log_prob)
                step_ref_type_reward = ((ref_type_sample.cpu() == (inst_type-1)).float() - 0.5) * 2
                self.ref_type_rewards.append(step_ref_type_reward)
            else:
                ref_type_sample = (inst_type - 1).cuda()


            # sample color
            color_probs = F.softmax(self.fc_color(inst_embedding), dim=1)
            color_sample, color_log_prob = sample_probs(color_probs)
            self.color_log_probs.append(color_log_prob)

            # sample shape
            shape_probs = F.softmax(self.fc_shape(inst_embedding), dim=1)
            shape_sample, shape_log_prob = sample_probs(shape_probs)
            self.shape_log_probs.append(shape_log_prob)

            if self.pred_act:
                # sample act
                act_probs = F.softmax(self.fc_act(inst_embedding), dim=1)
                act_sample, act_log_prob = sample_probs(act_probs)
                self.act_log_probs.append(act_log_prob)
                step_act_reward = ((act_sample.cpu() == gt_act).float() - 0.5) * 2
                self.act_rewards.append(step_act_reward)
            else:
                act_sample = gt_act.cuda()

            abs_ref_index = None
            if (1 - ref_type_sample).sum() > 0:
                abs_ref_index = torch.nonzero(1 - ref_type_sample).squeeze()
            rel_ref_index = None
            if ref_type_sample.sum() > 0:
                rel_ref_index = torch.nonzero(ref_type_sample).squeeze()
            self.rel_ref_indices.append(rel_ref_index)
            if abs_ref_index is not None:
                abs_inst_embedding = torch.index_select(inst_embedding, 0, Variable(abs_ref_index))
                abs_loc_predict, abs_loc_log_prob = self.loc_abs_predict(abs_inst_embedding)
            if rel_ref_index is not None:
                rel_inst_embedding = torch.index_select(inst_embedding, 0, Variable(rel_ref_index))
                rel_canvas_updated = torch.index_select(canvas_updated, 0, rel_ref_index.cpu())
                rel_loc_predict, rel_loc_log_prob, rel_att_log_prob, rel_step_att_reward = \
                    self.loc_relative_predict(rel_inst_embedding, Variable(rel_canvas_updated.float().cuda()), ref_obj=None)
            if abs_ref_index is not None and rel_ref_index is not None:
                loc_predict = index_join(abs_loc_predict, rel_loc_predict, abs_ref_index, rel_ref_index)
                loc_log_prob = index_join_var(abs_loc_log_prob, rel_loc_log_prob, abs_ref_index, rel_ref_index)
            elif abs_ref_index is not None:
                loc_predict = abs_loc_predict
                loc_log_prob = abs_loc_log_prob
            elif rel_ref_index is not None:
                loc_predict = rel_loc_predict
                loc_log_prob = rel_loc_log_prob
            else:
                assert False
            if rel_ref_index is not None:
                self.att_log_probs.append(rel_att_log_prob)
            else:
                self.att_log_probs.append(None)
            # sample location
            step_att_reward = None
            # if self.is_abs_inst(inst_type, dialog_ix):
            #     loc_predict, loc_log_prob = self.loc_abs_predict(inst_embedding)
            #     self.att_log_probs.append(None)
            # else:
            #     loc_predict, loc_log_prob, att_log_prob, step_att_reward = \
            #         self.loc_relative_predict(inst_embedding, Variable(canvas_updated.float().cuda()), ref_obj=ref)
            #     self.att_log_probs.append(att_log_prob)
            self.loc_log_probs.append(loc_log_prob)

            # compute rewards
            step_loc_reward = torch.zeros(loc_predict.size(0))
            step_color_reward = torch.zeros(loc_predict.size(0))
            step_shape_reward = torch.zeros(loc_predict.size(0))
            target_reward = torch.zeros(loc_predict.size(0)).fill_(-1)
            for i in range(step_loc_reward.size(0)):
                if loc_predict[i] < 0 or loc_predict[i] >= 25:
                    step_loc_reward[i] = -1
                    continue
                if act_sample[i] == 0:
                    # when the act is add, it must predict an object in the final canvas
                    predict_target = final_canvas[i, loc_predict[i]]
                else:
                    # when the act is delete, it must predict and object in the initial canvas
                    predict_target = init_canvas[i, loc_predict[i]]
                if (predict_target[2:] == target[i, 2:]).all():
                    target_reward[i] = 1.0
                # if not (predict_target == target[i]).all():
                if predict_target.sum() < 0:
                    step_loc_reward[i] = -1
                else:
                    step_loc_reward[i] = 1
                    # only consider color and shape prediction when the act is add
                    if act_sample[i] == 0:
                        step_color_reward[i] = 1 if color_sample[i] == predict_target[0] else -1
                        step_shape_reward[i] = 1 if shape_sample[i] == predict_target[1] else -1
            self.target_rewards.append(target_reward)
            # # if it's a relative reference instruction,
            # # reward is considered only when previous canvas is correctly computed
            # if dialog_ix > 0:
            #     for canvas_ix in range(running_reward.size(0)):
            #         # if running_reward[canvas_ix] < 0 and step_loc_reward[canvas_ix] < 0:
            #         if running_reward[canvas_ix] < 0:
            #             step_loc_reward[canvas_ix] = 0
            #             step_color_reward[canvas_ix] = 0
            #             step_shape_reward[canvas_ix] = 0
            #             if step_att_reward is not None:
            #                 step_att_reward[canvas_ix] = 0

            # current_reward = (step_loc_reward > 0) & (step_color_reward > 0) & (step_shape_reward > 0)
            # for reward_ix in range(running_reward.size(0)):
            #     if running_reward[reward_ix] > 0 and current_reward[reward_ix] == 1:
            #         running_reward[reward_ix] = 1
            #     else:
            #         running_reward[reward_ix] = -1
            # self.running_reward = running_reward

            # if ref is not None:
            #     for canvas_ix in range(canvas_updated.size(0)):
            #         final_canvas1 = final_canvas[canvas_ix]
            #         canvas_updated1 = canvas_updated[canvas_ix]
            #         canvas_correct = False
            #         obj_indices = torch.nonzero(canvas_updated1.sum(dim=1) >= 0).squeeze()
            #         if obj_indices.sum() > 0:
            #             # and (torch.index_select(final_canvas1, 0, obj_indices).sum(dim=1) >= 0).all()
            #             obj_current = torch.index_select(canvas_updated1, 0, obj_indices)
            #             obj_final = torch.index_select(final_canvas1, 0, obj_indices)
            #             if (obj_current == obj_final).all():
            #                 canvas_correct = True
            #         if not canvas_correct:
            #             step_loc_reward[canvas_ix] = 0
            #             step_color_reward[canvas_ix] = 0
            #             step_shape_reward[canvas_ix] = 0
            #             step_att_reward[canvas_ix] = 0

            self.att_rewards.append(step_att_reward)
            self.loc_rewards.append(step_loc_reward)
            self.color_rewards.append(step_color_reward)
            self.shape_rewards.append(step_shape_reward)
            for i in range(step_loc_reward.size(0)):
                if eval:
                    loc = loc_predict[i]
                    if 0 <= loc < 25:
                        if act_sample[i] == 0:
                            canvas_updated[i, loc] = torch.LongTensor([color_sample[i], shape_sample[i], loc // 5, loc % 5])
                        elif act_sample[i] == 1:
                            canvas_updated[i, loc].fill_(-2)
                else:
                    if act_sample[i] == 0 and step_loc_reward[i] > 0 and step_color_reward[i] > 0 and step_shape_reward[i] > 0:
                        loc = loc_predict[i]
                        canvas_updated[i, loc, 0] = color_sample[i]
                        canvas_updated[i, loc, 1] = shape_sample[i]
                        canvas_updated[i, loc, 2] = loc // 5
                        canvas_updated[i, loc, 3] = loc % 5
                    elif act_sample[i] == 1 and step_loc_reward[i] > 0:
                        loc = loc_predict[i]
                        canvas_updated[i, loc].fill_(-2)
            # canvas_updated = current_canvas.long()
        success = (((canvas_updated == final_canvas).sum(dim=2) == 4).sum(dim=1) == 25).float()
        self.success_reward = (success - 0.5) * 2
        return success.mean()
            # if step_loc_reward.size(0) == 1:
            #     if step_loc_reward[0] < 0 or step_color_reward[0] < 0 or step_shape_reward[0] < 0:
            #         break

        # canvas_predict = final_canvas.new(final_canvas.size())
        # canvas_predict.fill_(-1)
        # for i in range(canvas_predict.size(0)):
        # #     assert torch.equal(final_canvas[i, target[i, 2] * 5 + target[i, 3]], target[i])
        #     loc = loc_sample[i]
        #     canvas_predict[i, loc, 0] = color_sample[i]
        #     canvas_predict[i, loc, 1] = shape_sample[i]
        #     canvas_predict[i, loc, 2] = loc // 5
        #     canvas_predict[i, loc, 3] = loc % 5
        # # for now just consider one step
        # rewards = (((torch.eq(canvas_predict, final_canvas).sum(dim=2) == 4).sum(dim=1) == 25).float() - 0.5) * 2
        # # rewards = (((torch.eq(canvas_predict[:, :, 2:], final_canvas[:, :, 2:]).sum(dim=2) == 2).sum(dim=1) == 25).float() - 0.5) * 2
        # return color_rewards, shape_rewards, loc_rewards
        # return canvas_predict, final_canvas


class Shape2DObjCriterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model_out, act_target, target_obj, ref_obj):
        color_target = target_obj[:, 0]
        shape_target = target_obj[:, 1]
        act_out, color_out, shape_out, row_out, col_out = model_out
        loss = F.nll_loss(act_out, act_target) + \
               F.nll_loss(color_out, color_target) + \
               F.nll_loss(shape_out, shape_target) + \
               F.l1_loss(row_out, target_obj[:, 2].float()) + \
               F.l1_loss(col_out, target_obj[:, 3].float())
        # loss = F.nll_loss(color_out, color_target) + \
        #        F.nll_loss(shape_out, shape_target)
        rewards = F.l1_loss(row_out, target_obj[:, 2].float(), reduce=False).data + \
                  F.l1_loss(col_out, target_obj[:, 3].float(), reduce=False).data
        rewards = - rewards
        # model.rewards = rewards - model.running_baseline
        # model.running_baseline = 0.9 * model.running_baseline + 0.1 * rewards.mean()
        # model.rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-6)
        # model.rewards = (rewards - rewards.mean())
        rewards = (rewards - rewards.mean())
        return loss, rewards

def get_painter_model_prediction(painter_model, inst_samples, prev_canvas):
    assert inst_samples.size(1) == 21
    # [a, b, c, 0, d, 0] -> [a, b, c, 0, 0, 0]
    masks = (inst_samples == 0)
    for i in range(inst_samples.size(0)):
        if masks[i].sum() > 0:
            index = torch.nonzero(masks[i])[0, 0]
            inst_samples[i, index:] = 0
    samples_input = torch.zeros(inst_samples.size(0), inst_samples.size(1) + 2).long()
    # [a, b, ...] -> [0, a, b, ...]
    samples_input[:, 1:inst_samples.size(1) + 1] = inst_samples
    samples_input = Variable(samples_input.cuda())
    # vars = [Variable(var.data, volatile=True) for var in vars]
    output = painter_model(samples_input, prev_canvas, ref_obj=None, target_obj=None)
    act = torch.max(output[0].data, dim=1)[1]
    prediction = [p.data for p in output[1:]]
    target_obj = torch.stack([torch.max(prediction[0], dim=1)[1],
                              torch.max(prediction[1], dim=1)[1],
                              torch.round(prediction[2]).long(),
                              torch.round(prediction[3]).long()], dim=1)
    return act, target_obj


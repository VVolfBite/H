package net.lab1024.sa.admin.module.system.SupportDisasterInfo.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import net.lab1024.sa.admin.module.system.SupportDisasterInfo.dao.SupportDisasterDao;
import net.lab1024.sa.admin.module.system.SupportDisasterInfo.domain.entity.SupportDisasterEntity;
import net.lab1024.sa.admin.module.system.SupportDisasterInfo.domain.form.SupportDisasterAddForm;
import net.lab1024.sa.admin.module.system.SupportDisasterInfo.domain.form.SupportDisasterQueryForm;
import net.lab1024.sa.admin.module.system.SupportDisasterInfo.domain.vo.SupportDisasterVO;
import net.lab1024.sa.admin.module.system.SupportDisasterInfo.manager.SupportDisasterManager;
import net.lab1024.sa.common.common.code.UserErrorCode;
import net.lab1024.sa.common.common.domain.PageResult;
import net.lab1024.sa.common.common.domain.ResponseDTO;
import net.lab1024.sa.common.common.util.SmartBeanUtil;
import net.lab1024.sa.common.common.util.SmartPageUtil;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.*;
import java.util.stream.Collectors;

/**
 * 生命线工程service
 *
 * @Author 1024创新实验室-主任: 卓大
 * @Date 2021-12-29 21:52:46
 * @Wechat zhuoda1024
 * @Email lab1024@163.com
 * @Copyright 1024创新实验室 （ https://1024lab.net ）
 */
@Service
public class SupportDisasterService {
    @Autowired
    private SupportDisasterDao supportDisasterDao;

    @Autowired
    private SupportDisasterManager supportDisasterManager;


    public SupportDisasterEntity getByID(String ID) {
        return supportDisasterDao.selectById(ID);
    }


    /**
     * 查询生命线工程列表
     *
     * @param supportDisasterQueryForm
     * @return
     */
    public ResponseDTO<PageResult<SupportDisasterVO>> querySupportDisaster(SupportDisasterQueryForm supportDisasterQueryForm) {
        supportDisasterQueryForm.setDeletedFlag(false);
        Page pageParam = SmartPageUtil.convert2PageQuery(supportDisasterQueryForm);


        List<SupportDisasterVO> supportDisasterList = supportDisasterDao.querySupportDisaster(pageParam, supportDisasterQueryForm);
        if (CollectionUtils.isEmpty(supportDisasterList)) {
            PageResult<SupportDisasterVO> PageResult = SmartPageUtil.convert2PageResult(pageParam, supportDisasterList);
            return ResponseDTO.ok(PageResult);
        }

        List<String> supportDisasterIdList = supportDisasterList.stream().map(SupportDisasterVO::getID).collect(Collectors.toList());

        PageResult<SupportDisasterVO> PageResult = SmartPageUtil.convert2PageResult(pageParam, supportDisasterList);
        return ResponseDTO.ok(PageResult);
    }

    /**
     * 新增生命线工程
     *
     * @param supportDisasterAddForm
     * @return
     */
    public synchronized ResponseDTO<String> addSupportDisaster(SupportDisasterAddForm supportDisasterAddForm) {
        // 校验名称是否重复
        SupportDisasterEntity supportDisasterEntity = supportDisasterDao.getByID(supportDisasterAddForm.getID());
        if (null != supportDisasterEntity) {
            return ResponseDTO.userErrorParam("编码重复");
        }

        SupportDisasterEntity entity = SmartBeanUtil.copy(supportDisasterAddForm, SupportDisasterEntity.class);


        // 保存数据
        entity.setDeletedFlag(Boolean.FALSE);
        supportDisasterManager.saveSupportDisaster(entity);

        return ResponseDTO.ok();
    }

    /**
     * 更新生命线工程
     *
     * @param supportDisasterUpdateForm
     * @return
     */
    public synchronized ResponseDTO<String> updateSupportDisaster(SupportDisasterAddForm supportDisasterUpdateForm) {

        String ID = supportDisasterUpdateForm.getID();
        SupportDisasterEntity supportDisasterEntity = supportDisasterDao.selectById(ID);
        if (null == supportDisasterEntity) {
            return ResponseDTO.error(UserErrorCode.DATA_NOT_EXIST);
        }

        SupportDisasterEntity entity = SmartBeanUtil.copy(supportDisasterUpdateForm, SupportDisasterEntity.class);

        // 更新数据
        supportDisasterManager.updateSupportDisaster(entity);

        return ResponseDTO.ok();
    }

    /**
     * 批量删除生命线工程
     *
     * @param IdList 生命线工程ID列表
     * @return
     */
    public ResponseDTO<String> batchUpdateDeleteFlag(List<String> IdList) {
        if (CollectionUtils.isEmpty(IdList)) {
            return ResponseDTO.ok();
        }
        List<SupportDisasterEntity> supportDisasterEntityList = supportDisasterManager.listByIds(IdList);
        if (CollectionUtils.isEmpty(supportDisasterEntityList)) {
            return ResponseDTO.ok();
        }
        // 更新删除
        List<SupportDisasterEntity> deleteList = IdList.stream().map(e -> {
            SupportDisasterEntity updateSupportDisaster = new SupportDisasterEntity();
            updateSupportDisaster.setID(e);
            updateSupportDisaster.setDeletedFlag(true);
            return updateSupportDisaster;
        }).collect(Collectors.toList());
        supportDisasterManager.updateBatchById(deleteList);

        return ResponseDTO.ok();
    }


    /**
     * 查询全部员工
     *
     * @return
     */
    public ResponseDTO<List<SupportDisasterVO>> queryAllSupportDisaster() {
        List<SupportDisasterVO> supportDisasterList = supportDisasterDao.selectSupportDisasterByDeleted(Boolean.FALSE);
        return ResponseDTO.ok(supportDisasterList);
    }

    /**
     * 根据发生的地点获取员工
     *
     * @param location
     * @return
     */
    public SupportDisasterEntity getByLocation(String location) {
        return supportDisasterDao.getByLocation(location);
    }

}



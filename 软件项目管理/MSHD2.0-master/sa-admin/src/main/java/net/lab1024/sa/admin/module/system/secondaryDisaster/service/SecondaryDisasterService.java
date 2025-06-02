package net.lab1024.sa.admin.module.system.secondaryDisaster.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import net.lab1024.sa.admin.module.system.secondaryDisaster.dao.SecondaryDisasterDao;
import net.lab1024.sa.admin.module.system.secondaryDisaster.domain.entity.SecondaryDisasterEntity;
import net.lab1024.sa.admin.module.system.secondaryDisaster.domain.form.SecondaryDisasterAddForm;
import net.lab1024.sa.admin.module.system.secondaryDisaster.domain.form.SecondaryDisasterQueryForm;
import net.lab1024.sa.admin.module.system.secondaryDisaster.domain.vo.SecondaryDisasterVO;
import net.lab1024.sa.admin.module.system.secondaryDisaster.manager.SecondaryDisasterManager;
import net.lab1024.sa.common.common.code.UserErrorCode;
import net.lab1024.sa.common.common.domain.PageResult;
import net.lab1024.sa.common.common.domain.ResponseDTO;
import net.lab1024.sa.common.common.util.SmartBeanUtil;
import net.lab1024.sa.common.common.util.SmartPageUtil;
import org.apache.commons.collections4.CollectionUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import java.util.List;
import java.util.stream.Collectors;

/**
 * ClassName: SecondaryDisasterService
 * Package: net.lab1024.sa.admin.module.system.secondaryDisaster.service
 * Description:
 *
 * @Author 幻秋
 * @Create 2023/11/5  20:40
 * @Version 1.0
 */
@Service
public class SecondaryDisasterService {
    @Autowired
    private SecondaryDisasterDao secondaryDisasterDao;

    @Autowired
    private SecondaryDisasterManager secondaryDisasterManager;


    public SecondaryDisasterEntity getByID(String ID) {
        return secondaryDisasterDao.selectById(ID);
    }


    /**
     * 查询次生灾害列表
     *
     * @param secondaryDisasterQueryForm
     * @return
     */
    public ResponseDTO<PageResult<SecondaryDisasterVO>> querySecondaryDisaster(SecondaryDisasterQueryForm secondaryDisasterQueryForm) {
        secondaryDisasterQueryForm.setDeletedFlag(false);
        Page pageParam = SmartPageUtil.convert2PageQuery(secondaryDisasterQueryForm);


        List<SecondaryDisasterVO> secondaryDisasterList = secondaryDisasterDao.querySecondaryDisaster(pageParam, secondaryDisasterQueryForm);
        if (CollectionUtils.isEmpty(secondaryDisasterList)) {
            PageResult<SecondaryDisasterVO> PageResult = SmartPageUtil.convert2PageResult(pageParam, secondaryDisasterList);
            return ResponseDTO.ok(PageResult);
        }

        List<String> secondaryDisasterIdList = secondaryDisasterList.stream().map(SecondaryDisasterVO::getID).collect(Collectors.toList());

        PageResult<SecondaryDisasterVO> PageResult = SmartPageUtil.convert2PageResult(pageParam, secondaryDisasterList);
        return ResponseDTO.ok(PageResult);
    }

    /**
     * 新增次生灾害
     *
     * @param secondaryDisasterAddForm
     * @return
     */
    public synchronized ResponseDTO<String> addSecondaryDisaster(SecondaryDisasterAddForm secondaryDisasterAddForm) {
        // 校验名称是否重复
        SecondaryDisasterEntity secondaryDisasterEntity = secondaryDisasterDao.getByID(secondaryDisasterAddForm.getID());
        if (null != secondaryDisasterEntity) {
            return ResponseDTO.userErrorParam("编码重复");
        }

        SecondaryDisasterEntity entity = SmartBeanUtil.copy(secondaryDisasterAddForm, SecondaryDisasterEntity.class);


        // 保存数据
        entity.setDeletedFlag(Boolean.FALSE);
        secondaryDisasterManager.saveSecondaryDisaster(entity);

        return ResponseDTO.ok();
    }

    /**
     * 更新次生灾害
     *
     * @param secondaryDisasterUpdateForm
     * @return
     */
    public synchronized ResponseDTO<String> updateSecondaryDisaster(SecondaryDisasterAddForm secondaryDisasterUpdateForm) {

        String ID = secondaryDisasterUpdateForm.getID();
        SecondaryDisasterEntity secondaryDisasterEntity = secondaryDisasterDao.selectById(ID);
        if (null == secondaryDisasterEntity) {
            return ResponseDTO.error(UserErrorCode.DATA_NOT_EXIST);
        }

        SecondaryDisasterEntity entity = SmartBeanUtil.copy(secondaryDisasterUpdateForm, SecondaryDisasterEntity.class);

        // 更新数据
        secondaryDisasterManager.updateSecondaryDisaster(entity);

        return ResponseDTO.ok();
    }

    /**
     * 批量删除次生灾害
     *
     * @param IdList 生次生灾害ID列表
     * @return
     */
    public ResponseDTO<String> batchUpdateDeleteFlag(List<String> IdList) {
        if (CollectionUtils.isEmpty(IdList)) {
            return ResponseDTO.ok();
        }
        List<SecondaryDisasterEntity> secondaryDisasterEntityList = secondaryDisasterManager.listByIds(IdList);
        if (CollectionUtils.isEmpty(secondaryDisasterEntityList)) {
            return ResponseDTO.ok();
        }
        // 更新删除
        List<SecondaryDisasterEntity> deleteList = IdList.stream().map(e -> {
            SecondaryDisasterEntity updateSecondaryDisaster = new SecondaryDisasterEntity();
            updateSecondaryDisaster.setID(e);
            updateSecondaryDisaster.setDeletedFlag(true);
            return updateSecondaryDisaster;
        }).collect(Collectors.toList());
        secondaryDisasterManager.updateBatchById(deleteList);

        return ResponseDTO.ok();
    }


    /**
     * 查询全部员工
     *
     * @return
     */
    public ResponseDTO<List<SecondaryDisasterVO>> queryAllSecondaryDisaster() {
        List<SecondaryDisasterVO> secondaryDisasterList = secondaryDisasterDao.selectSecondaryDisasterByDeleted(Boolean.FALSE);
        return ResponseDTO.ok(secondaryDisasterList);
    }

    /**
     * 根据发生的地点获取员工
     *
     * @param location
     * @return
     */
    public SecondaryDisasterEntity getByLocation(String location) {
        return secondaryDisasterDao.getByLocation(location);
    }

}

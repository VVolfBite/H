package net.lab1024.sa.admin.module.system.CasualtyInfo.service;

import com.baomidou.mybatisplus.extension.plugins.pagination.Page;
import net.lab1024.sa.admin.module.system.CasualtyInfo.dao.CasualtyDao;
import net.lab1024.sa.admin.module.system.CasualtyInfo.domain.entity.CasualtyEntity;
import net.lab1024.sa.admin.module.system.CasualtyInfo.domain.form.CasualtyAddForm;
import net.lab1024.sa.admin.module.system.CasualtyInfo.domain.form.CasualtyQueryForm;
import net.lab1024.sa.admin.module.system.CasualtyInfo.domain.vo.CasualtyVO;
import net.lab1024.sa.admin.module.system.CasualtyInfo.manager.CasualtyManager;
import net.lab1024.sa.admin.module.system.SupportDisasterInfo.domain.entity.SupportDisasterEntity;
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

@Service
public class CasualtyService {

    @Autowired
    private CasualtyDao casualtyDao;


    @Autowired
    private CasualtyManager casualtyManager;


    public CasualtyEntity getById(String ID){return casualtyDao.selectById(ID);}


    /**
     * 查询伤亡列表
     * @param casualtyQueryForm
     * @return
     */
    public ResponseDTO<PageResult<CasualtyVO>> queryCasualty(CasualtyQueryForm casualtyQueryForm){
       casualtyQueryForm.setDeletedFlag(false);
       Page pageParam = SmartPageUtil.convert2PageQuery(casualtyQueryForm);

        List<CasualtyVO> casualtyList = casualtyDao.queryCasualty(pageParam,casualtyQueryForm);
        if(CollectionUtils.isEmpty(casualtyList)){
            PageResult<CasualtyVO> PageResult = SmartPageUtil.convert2PageResult(pageParam,casualtyList);
            return ResponseDTO.ok(PageResult);
        }
        List<String> casualtyIdList = casualtyList.stream().map(CasualtyVO::getID).collect(Collectors.toList());

        PageResult<CasualtyVO> PageResult = SmartPageUtil.convert2PageResult(pageParam,casualtyList);
        return ResponseDTO.ok(PageResult);
    }

    /**
     * 新增伤亡信息
     * @param casualtyAddForm
     * @return
     */
    public  synchronized ResponseDTO<String> addCasualty(CasualtyAddForm casualtyAddForm){
        CasualtyEntity casualtyEntity = casualtyDao.getByID(casualtyAddForm.getID());
        if (null != casualtyEntity){
            return ResponseDTO.userErrorParam("编码错误");
        }
        CasualtyEntity entity = SmartBeanUtil.copy(casualtyAddForm, CasualtyEntity.class);

        // 保存数据
        entity.setDeletedFlag(Boolean.FALSE);
        casualtyManager.saveCasualty(entity);

        return ResponseDTO.ok();
    }

    /**
     * 更新伤亡信息
     * @param casualtyUpdateForm
     * @return
     */
    public synchronized ResponseDTO<String> updateCasualty(CasualtyAddForm casualtyUpdateForm){
        String ID = casualtyUpdateForm.getID();
        CasualtyEntity casualtyEntity = casualtyDao.selectById(ID);
        if(null == casualtyEntity){
            return ResponseDTO.error(UserErrorCode.DATA_NOT_EXIST);
        }

        CasualtyEntity entity = SmartBeanUtil.copy(casualtyUpdateForm, CasualtyEntity.class);

        // 更新数据
        casualtyManager.updateCasualty(entity);

        return  ResponseDTO.ok();
    }

    /**
     * 批量删除伤亡信息
     * @param IdList
     * @return
     */
    public ResponseDTO<String> batchUpdateDeletedFlag(List<String> IdList){
        if(CollectionUtils.isEmpty(IdList)){
            return ResponseDTO.ok();
        }
        List<CasualtyEntity> casualtyEntityList = casualtyManager.listByIds(IdList);
        if (CollectionUtils.isEmpty(casualtyEntityList)){
            return  ResponseDTO.ok();
        }
        List<CasualtyEntity> deleteList = IdList.stream().map(e->{
            CasualtyEntity updateCasualty = new CasualtyEntity();
            updateCasualty.setID(e);
            updateCasualty.setDeletedFlag(true);
            return updateCasualty;
        }).collect(Collectors.toList());
        casualtyManager.updateBatchById(deleteList);

        return ResponseDTO.ok();
    }

    /**
     * 查询全部的伤害信息
     * @return
     */
    public ResponseDTO<List<CasualtyVO>> queryAllCasualty(){
        List<CasualtyVO> casualtyList = casualtyDao.selectCasualtyByDeleted(Boolean.FALSE);
        return ResponseDTO.ok(casualtyList);
    }

    /**
     * 根据发生地点获取伤害信息
     * @param location
     * @return
     */
    public CasualtyEntity getLocation(String location) {return casualtyDao.getByLocation(location);}

}

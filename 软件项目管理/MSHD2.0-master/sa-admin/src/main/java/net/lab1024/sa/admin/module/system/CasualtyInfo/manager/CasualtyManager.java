package net.lab1024.sa.admin.module.system.CasualtyInfo.manager;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import net.lab1024.sa.admin.module.system.CasualtyInfo.dao.CasualtyDao;
import net.lab1024.sa.admin.module.system.CasualtyInfo.domain.entity.CasualtyEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
public class CasualtyManager extends ServiceImpl<CasualtyDao, CasualtyEntity>{
    @Autowired
    private CasualtyDao casualtyDao;


    /**
     * 保存伤亡信息
     * @param casualtyEntity
     */
    @Transactional(rollbackFor = Throwable.class)
    public void saveCasualty(CasualtyEntity casualtyEntity){
        casualtyDao.insert(casualtyEntity);
    }

    /**
     * 更新伤亡信息
     * @param casualtyEntity
     */
    @Transactional(rollbackFor = Throwable.class)
    public void updateCasualty(CasualtyEntity casualtyEntity){
        casualtyDao.updateById(casualtyEntity);
    }
}

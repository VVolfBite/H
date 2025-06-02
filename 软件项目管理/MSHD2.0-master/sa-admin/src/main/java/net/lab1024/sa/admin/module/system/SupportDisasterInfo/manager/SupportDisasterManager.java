package net.lab1024.sa.admin.module.system.SupportDisasterInfo.manager;

import com.baomidou.mybatisplus.extension.service.impl.ServiceImpl;
import net.lab1024.sa.admin.module.system.SupportDisasterInfo.dao.SupportDisasterDao;
import net.lab1024.sa.admin.module.system.SupportDisasterInfo.domain.entity.SupportDisasterEntity;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;



@Service
public class SupportDisasterManager extends ServiceImpl<SupportDisasterDao, SupportDisasterEntity> {
    @Autowired
    private SupportDisasterDao supportDisasterDao;


    /**
     * 保存生命线工程
     *
     * @param supportDisaster
     */
    @Transactional(rollbackFor = Throwable.class)
    public void saveSupportDisaster(SupportDisasterEntity supportDisaster) {
        // 保存员工 获得id
        supportDisasterDao.insert(supportDisaster);
    }

    /**
     * 更新生命线工程
     *
     * @param supportDisaster
     */
    @Transactional(rollbackFor = Throwable.class)
    public void updateSupportDisaster(SupportDisasterEntity supportDisaster) {
        // 保存员工 获得id
        supportDisasterDao.updateById(supportDisaster);
    }

}
